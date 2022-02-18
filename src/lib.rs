//! A thread-safe object pool with automatic return and attach/detach semantics
//!
//! The goal of an object pool is to reuse expensive to allocate objects or frequently allocated objects
//!
//! # Examples
//!
//! ## Creating a Pool
//!
//! The general pool creation looks like this
//! ```
//!  let pool: Pool<T> = Pool::new(capacity, || T::new());
//! ```
//! Example pool with 32 `Vec<u8>` with capacity of 4096
//! ```
//!  let pool: Pool<Vec<u8>> = Pool::new(32, || Vec::with_capacity(4096));
//! ```
//!
//! ## Using a Pool
//!
//! Basic usage for pulling from the pool
//! ```
//! let pool: Pool<Vec<u8>> = Pool::new(32, || Vec::with_capacity(4096));
//! let mut reusable_buff = pool.pull().unwrap(); // returns None when the pool is saturated
//! reusable_buff.clear(); // clear the buff before using
//! some_file.read_to_end(reusable_buff);
//! // reusable_buff is automatically returned to the pool when it goes out of scope
//! ```
//! Pull from pool and `detach()`
//! ```
//! let pool: Pool<Vec<u8>> = Pool::new(32, || Vec::with_capacity(4096));
//! let mut reusable_buff = pool.pull().unwrap(); // returns None when the pool is saturated
//! reusable_buff.clear(); // clear the buff before using
//! let (pool, reusable_buff) = reusable_buff.detach();
//! let mut s = String::from(reusable_buff);
//! s.push_str("hello, world!");
//! pool.attach(s.into_bytes()); // reattach the buffer before reusable goes out of scope
//! // reusable_buff is automatically returned to the pool when it goes out of scope
//! ```
//!
//! ## Using Across Threads
//!
//! You simply wrap the pool in a [`std::sync::Arc`]
//! ```
//! let pool: Arc<Pool<T>> = Arc::new(Pool::new(cap, || T::new()));
//! ```
//!
//! # Warning
//!
//! Objects in the pool are not automatically reset, they are returned but NOT reset
//! You may want to call `object.reset()` or  `object.clear()`
//! or any other equivalent for the object that you are using, after pulling from the pool
//!
//! [`std::sync::Arc`]: https://doc.rust-lang.org/stable/std/sync/struct.Arc.html

use parking_lot::Mutex;
use std::mem::{forget, ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

pub type Stack<T> = Vec<T>;

pub struct Pool<T> {
    objects: Mutex<Stack<T>>,
    pub name: String,
    pub last_fail: Mutex<Instant>,
    pub cnt_fail: AtomicUsize,
}

impl<T> Pool<T> {
    #[inline]
    pub fn new<F>(name: String, cap: usize, init: F) -> Pool<T>
    where
        F: Fn() -> T,
    {
        let mut objects = Stack::new();

        for _ in 0..cap {
            objects.push(init());
        }

        Pool {
            objects: Mutex::new(objects),
            name,
            last_fail: Mutex::new(Instant::now()),
            cnt_fail: AtomicUsize::new(0),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.objects.lock().len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.objects.lock().is_empty()
    }

    #[inline]
    pub fn attach(&self, t: T) {
        self.objects.lock().push(t)
    }
}

pub fn try_pull<T>(pool: Arc<Pool<T>>) -> Option<Reusable<T>> {
    pool.objects
        .lock()
        .pop()
        .map(|data| Reusable::new(Some(pool.clone()), data))
}

pub fn pull<T, F: Fn() -> T>(pool: Arc<Pool<T>>, fallback: F) -> Reusable<T> {
    try_pull(pool.clone()).unwrap_or_else(|| Reusable::new(Some(pool.clone()), fallback()))
}

pub struct Reusable<T> {
    pool: Option<Arc<Pool<T>>>,
    data: ManuallyDrop<T>,
}

impl<T> Reusable<T> {
    #[inline]
    pub fn new(pool: Option<Arc<Pool<T>>>, t: T) -> Self {
        Self {
            pool,
            data: ManuallyDrop::new(t),
        }
    }

    #[inline]
    pub fn detach(mut self) -> (Option<Arc<Pool<T>>>, T) {
        let pool = self.pool.clone();
        let ret = unsafe { (pool, self.take()) };
        forget(self);
        ret
    }

    unsafe fn take(&mut self) -> T {
        ManuallyDrop::take(&mut self.data)
    }
}

impl<T> Deref for Reusable<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for Reusable<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> Drop for Reusable<T> {
    #[inline]
    fn drop(&mut self) {
        if let Some(p) = self.pool.as_mut() {
            let pool = p.clone();
            unsafe { pool.attach(self.take()) }
        } else {
            unsafe {
                ManuallyDrop::drop(&mut self.data);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{pull, try_pull, Pool, Reusable};
    use std::mem::drop;
    use std::sync::Arc;

    #[test]
    fn detach() {
        let pool = Arc::new(Pool::new("test".to_string(), 1, || Vec::new()));
        let (pool, mut object) = try_pull(pool).unwrap().detach();
        object.push(1);
        Reusable::new(pool.clone(), object);
        assert_eq!(try_pull(pool.unwrap()).unwrap()[0], 1);
    }

    #[test]
    fn detach_then_attach() {
        let pool = Arc::new(Pool::new("test".to_string(), 1, || Vec::new()));
        let (pool, mut object) = try_pull(pool).unwrap().detach();
        object.push(1);
        pool.as_ref().unwrap().attach(object);
        assert_eq!(try_pull(pool.unwrap()).unwrap()[0], 1);
    }

    #[test]
    fn test_pull() {
        let pool = Arc::new(Pool::<Vec<u8>>::new("test".to_string(), 1, || Vec::new()));

        let object1 = try_pull(pool.clone());
        let object2 = try_pull(pool.clone());
        let object3 = pull(pool.clone(), || Vec::new());

        assert!(object1.is_some());
        assert!(object2.is_none());
        drop(object1);
        drop(object2);
        drop(object3);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn e2e() {
        let pool = Arc::new(Pool::new("test".to_string(), 10, || Vec::new()));
        let mut objects = Vec::new();

        for i in 0..10 {
            let mut object = try_pull(pool.clone()).unwrap();
            object.push(i);
            objects.push(object);
        }

        assert!(try_pull(pool.clone()).is_none());
        drop(objects);
        assert!(try_pull(pool.clone()).is_some());

        for i in 10..0 {
            let mut object = pool.objects.lock().pop().unwrap();
            assert_eq!(object.pop(), Some(i));
        }
    }
}
