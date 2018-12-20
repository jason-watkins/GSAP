// Copyright (c) 2018 United States Government as represented by the
// Administrator of the National Aeronautics and Space Administration.
// All Rights Reserved.
#ifndef PCOE_LOCK_H
#define PCOE_LOCK_H
#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

#include "Contracts.h"

namespace PCOE {
    /**
     * Provides a fast, lightweight locking primitive that avoids the overhead
     * of an OS provided mutex when possible.
     *
     * @remarks
     * In the case where a lock is uncontested, locks and unlocks amount to a
     * single compare/exchange operation each. If a thread fails to immediately
     * aquire the lock, it will fall back to the "slow" path, where it will spin
     * several times waiting for the lock to become available. If it is still
     * unable to aquire the lock, the thread will fall back to waiting on an OS
     * mutex.
     *
     * @remarks
     * Since any given thread can only be waiting on a single lock at any given
     * time, the classes provided by this module allocate one OS mutex per
     * thread, rather than one per lock. This strategy keeps each individual
     * lock as small as possible (a single pointer size), while introducing
     * minimal overhead for each aditional thread. Each thread has thread-local
     * storage for a single pointer allocated when the thread is created. If the
     * thread ever tries to aquire a lock and is forced onto the slow path, An
     * aditional struct will be allocated with an OS mutex and related
     * bookkeeping memory for the thread.
     *
     * @remarks
     * This class satisfies the {@code Mutex} named requirement.
     *
     * @remarks
     * This class was inspired by
     * https://webkit.org/blog/6161/locking-in-webkit/
     *
     * @author Jason Watkins
     **/
    class Lock final {
    public:
        /**
         * Constructs a new {@code Lock} in an unlocked state.
         **/
        constexpr Lock() noexcept : state(0) {}

        /**
         * Deleted copy constructor
         **/
        Lock(const Lock&) = delete;

        /**
         * Deleted move constructor
         **/
        Lock(Lock&&) = delete;

        /**
         * Releases all resources associated with the mutex.
         **/
        ~Lock() {
            Expect(state.load() == 0, "Destroying locked mutex");
        }

        /**
         * Aquires the lock. If another thread already owns the lock, blocks
         * execution of the current thread until the lock can be aquired.
         *
         * @remarks
         * If this method is called by a thread that already owns the lock,
         * behavior is undefined. The most likely outcome is that the thread
         * will deadlock.
         *
         * @remarks
         * It is usually not necessary to call @{code lock} directly. use
         * {@code std::unique_lock} and {@code std::lock_guard} to manage
         * locking.
         **/
        inline void lock() {
            state_type expected = 0;
            state_type desired = locked_bit;
            if (state.compare_exchange_strong(expected, desired, std::memory_order_acquire)) {
                return;
            }

            lock_slow();
        }

        /**
         * Attempts to aquire the lock without blocking.
         *
         * @remarks
         * This method does not guarantee that it will aquire the lock if it is
         * available. It may return false even if no other thread currently owns
         * the lock.
         *
         * @remarks
         * If this method is called by a thread that already owns the lock,
         * behavior is undefined.
         *
         * @return true if the lock was aquired; otherwise, false.
         **/
        inline bool try_lock() noexcept {
            state_type value = state.load();
            state_type expected = value & ~locked_bit;
            state_type desired = value | locked_bit;
            return state.compare_exchange_strong(expected, desired, std::memory_order_acquire);
        }

        /**
         * Releases the lock.
         *
         * @remarks
         * If the current thread does not own the lock, behavior is undefined.
         *
         * @remarks
         * It is usually not necessary to call @{code unlock} directly. use
         * {@code std::unique_lock} and {@code std::lock_guard} to manage
         * locking.
         **/
        inline void unlock() {
            state_type expected = locked_bit;
            state_type desired = 0;
            if (state.compare_exchange_strong(expected, desired, std::memory_order_release)) {
                return;
            }

            unlock_slow();
        }

    private:
        void lock_slow();

        void unlock_slow();

        using state_type = std::uintptr_t;

        const state_type locked_bit = 0x01;
        const state_type queue_locked_bit = 0x02;
        const state_type queue_mask = ~(locked_bit | queue_locked_bit);
        std::atomic<state_type> state = 0x00;
    };

    /**
     * Provides a fast, lightweight locking primitive that avoids the overhead
     * of an OS provided mutex when possible. The {@code TimedLock} expands on
     * the {@code Lock} class by providing functions for locking with a timeout.
     *
     * @remarks
     * In the case where a lock is uncontested, locks and unlocks amount to a
     * single compare/exchange operation each. If a thread fails to immediately
     * aquire the lock, it will fall back to the "slow" path, where it will spin
     * several times waiting for the lock to become available. If it is still
     * unable to aquire the lock, the thread will fall back to waiting on an OS
     * mutex. In the case where lock is called via one of the timed versions,
     * the thread is guaranteed to try to aquire the lock *at least* until the
     * timeout has elapsed. The lock will make its best effort to fail
     * immediately, but the only guarantee is that it will return some time
     * after the timeout has elapsed if it is unable to aquire the lock
     * earilier.
     *
     * @remarks
     * Since any given thread can only be waiting on a single lock at any given
     * time, the classes provided by this module allocate one OS mutex per
     * thread, rather than one per lock. This strategy keeps each individual
     * lock as small as possible (a single pointer size), while introducing
     * minimal overhead for each aditional thread. Each thread has thread-local
     * storage for a single pointer allocated when the thread is created. If the
     * thread ever tries to aquire a lock and is forced onto the slow path, An
     * aditional struct will be allocated with an OS mutex and related
     * bookkeeping memory for the thread.
     *
     * @remarks
     * This class satisfies the {@code TimedMutex} named requirement.
     *
     * @remarks
     * This class was inspired by
     * https://webkit.org/blog/6161/locking-in-webkit/
     *
     * @author Jason Watkins
     **/
    class TimedLock {
    public:
        /**
         * Aquires the lock. If another thread already owns the lock, blocks
         * execution of the current thread until the lock can be aquired.
         *
         * @remarks
         * If this method is called by a thread that already owns the lock,
         * behavior is undefined. The most likely outcome is that the thread
         * will deadlock.
         *
         * @remarks
         * It is usually not necessary to call @{code lock} directly. use
         * {@code std::unique_lock} and {@code std::lock_guard} to manage
         * locking.
         **/
        inline void lock() {
            state_type expected = 0;
            state_type desired = locked_bit;
            if (state.compare_exchange_strong(expected, desired, std::memory_order_acquire)) {
                return;
            }

            bool locked;
            do {
                locked = try_lock_slow(std::chrono::microseconds::max());
            } while (!locked);
        }

        /**
         * Attempts to aquire the lock without blocking.
         *
         * @remarks
         * This method does not guarantee that it will aquire the lock if it is
         * available. It may return false even if no other thread currently owns
         * the lock.
         *
         * @remarks
         * If this method is called by a thread that already owns the lock,
         * behavior is undefined.
         *
         * @return true if the lock was aquired; otherwise, false.
         **/
        inline bool try_lock() {
            state_type value = state.load();
            state_type expected = value & ~locked_bit;
            state_type desired = value | locked_bit;
            return state.compare_exchange_strong(expected, desired, std::memory_order_acquire);
        }

        /**
         * Attempts to aquire the lock. If another thread already owns the lock,
         * blocks execution of the current thread until the lock can be aquired
         * or until the specified duration has elapsed.
         *
         * @remarks
         * If {@p dur} is less than or equal to dur.zero(), this function
         * behaves like {@code try_lock}. This function may continue blocking
         * after the timeout duration has elapsed due to scheduling or resource
         * contention delays.
         *
         * @remarks
         * If this method is called by a thread that already owns the lock,
         * behavior is undefined.
         *
         * @remarks
         * It is usually not necessary to call @{code try_lock_for} directly.
         * use {@code std::unique_lock} to manage locking with timeouts.
         *
         * @param dur The minimum time the thread should block while waiting for
         *            the lock.
         * @return    true if the lock was aquired; otherwise, false.
         **/
        template <class Rep, class Period = std::ratio<1>>
        inline bool try_lock_for(const std::chrono::duration<Rep, Period>& dur) {
            using namespace std::chrono;

            if (try_lock()) {
                return true;
            }
            auto micros = duration_cast<microseconds>(dur);
            return try_lock_slow(micros);
        }

        /**
         * Attempts to aquire the lock. If another thread already owns the lock,
         * blocks execution of the current thread until the lock can be aquired
         * or until the specified timeout time is reached.
         *
         * @remarks
         * If {@p time} has already passed, this function behaves like
         * {@code try_lock}. This function may continue blocking after the
         * timeout time has been reached due to scheduling or resource
         * contention delays.
         *
         * @remarks
         * If this method is called by a thread that already owns the lock,
         * behavior is undefined.
         *
         * @remarks
         * It is usually not necessary to call @{code try_lock_until} directly.
         * use {@code std::unique_lock} to manage locking with timeouts.
         *
         * @param time The earlierst time at which the thread should stop trying
         *             to aquire the lock.
         * @return     true if the lock was aquired; otherwise, false.
         **/
        template <class Clock, class Duration = typename Clock::duration>
        inline bool try_lock_until(const std::chrono::time_point<Clock, Duration>& time) {
            using namespace std::chrono;

            if (try_lock()) {
                return true;
            }

            auto now = Clock::now();
            if (now > time) {
                return false;
            }
            auto micros = duration_cast<microseconds>(time - now);
            return try_lock_slow(micros);
        }

        /**
         * Releases the lock.
         *
         * @remarks
         * If the current thread does not own the lock, behavior is undefined.
         *
         * @remarks
         * It is usually not necessary to call @{code unlock} directly. use
         * {@code std::unique_lock} and {@code std::lock_guard} to manage
         * locking.
         **/
        inline void unlock() {
            state_type expected = locked_bit;
            state_type desired = 0;
            if (state.compare_exchange_strong(expected, desired, std::memory_order_release)) {
                return;
            }

            unlock_slow();
        }

    private:
        bool try_lock_slow(std::chrono::microseconds micros);

        void unlock_slow();

        using state_type = std::uintptr_t;
        state_type locked_bit = 0x01;
        state_type queue_locked_bit = 0x02;
        state_type queue_mask = ~(locked_bit | queue_locked_bit);
        std::atomic<state_type> state = 0x00;
    };

    /**
     * Provides a fast, lightweight locking primitive that avoids the overhead
     * of an OS provided mutex when possible. The {@code RecursiveLock} expands
     * on the {@code Lock} class by providing making the lock re-entrant. i.e.,
     * a thread that owns the lock can safely call any variant of {@code lock}
     * again without the risk of a deadlock. The aditional overhead of timing is
     * judged to be minimal compared to the overhead of recursion, so all
     * recursive locks also provide the option of locking with a timeout.
     *
     * @remarks
     * A recursive lock has to do substantially more bookkeeping that a
     * non-recursive lock. In the case where a thread already owns the lock and
     * is entering or leaving recursively, locks and unlocks amount to an atomic
     * load and an atomic fetch/add each. In the case where a thread is
     * aquiring an uncontested lock for the first time, the thread must perform
     * several atomic operations. All other cases will fall back to the "slow"
     * path, where the thread will spin several times waiting for the lock to
     * become available, and finally fall back to waiting on an OS mutex.
     *
     * @remarks
     * Since any given thread can only be waiting on a single lock at any given
     * time, the classes provided by this module allocate one OS mutex per
     * thread, rather than one per lock. This strategy keeps each individual
     * lock as small as possible (a single pointer size), while introducing
     * minimal overhead for each aditional thread. Each thread has thread-local
     * storage for a single pointer allocated when the thread is created. If the
     * thread ever tries to aquire a lock and is forced onto the slow path, An
     * aditional struct will be allocated with an OS mutex and related
     * bookkeeping memory for the thread.
     *
     * @remarks
     * This class satisfies the {@code TimedMutex} named requirement.
     *
     * @remarks
     * This class was inspired by
     * https://webkit.org/blog/6161/locking-in-webkit/
     *
     * @author Jason Watkins
     **/
    class RecursiveLock {
    public:
        /**
         * Aquires the lock. If another thread already owns the lock, blocks
         * execution of the current thread until the lock can be aquired. A
         * thread may aquire the same {@code RecursiveLock} repeatedly. The lock
         * will not be released until a matching number of calls to
         * {@code unlock} are made.
         *
         * @remarks
         * It is usually not necessary to call @{code lock} directly. use
         * {@code std::unique_lock} and {@code std::lock_guard} to manage
         * locking.
         **/
        inline void lock() {
            bool locked = try_lock();
            while (!locked) {
                locked = try_lock_slow(std::chrono::microseconds::max());
            }
        }

        /**
         * Attempts to aquire the lock without blocking. A thread may aquire the
         * same {@code RecursiveLock} repeatedly. The lock will not be released
         * until a matching number of calls to {@code unlock} are made.
         *
         * @remarks
         * This method does not guarantee that it will aquire the lock if it is
         * available. It may return false even if no other thread currently owns
         * the lock.
         *
         * @return true if the lock was aquired; otherwise, false.
         **/
        inline bool try_lock() {
            if (owner_id.load() == std::this_thread::get_id()) {
                // We already own the lock, so we just need to increment the
                // count.
                state_type prev = state.fetch_add(1, std::memory_order_acquire);
                Ensure(prev & count_mask > 0, "Lock not held while incrementing recursion count");
                Ensure(prev < count_mask, "Recursive lock overflow");
                return true;
            }
            else {
                // If the state lock is free and the count is 0, then no one
                // owns the lock and we're free to take it. Try to set the count
                // to 1 and aquire the state lock. If we succeed, set the owner
                // id and release the state lock.
                state_type expected = 0x0000;
                state_type desired = 0x0001 | state_locked_bit;
                if (state.compare_exchange_strong(expected, desired, std::memory_order_acquire)) {
                    owner_id = std::this_thread::get_id();
                    state.fetch_sub(state_locked_bit, std::memory_order_release);
                    return true;
                }
            }
            return false;
        }

        /**
         * Attempts to aquire the lock. If another thread already owns the lock,
         * blocks execution of the current thread until the lock can be aquired
         * or until the specified duration has elapsed. A thread may aquire the
         * same {@code RecursiveLock} repeatedly. The lock will not be released
         * until a matching number of calls to {@code unlock} are made.
         *
         * @remarks
         * If {@p dur} is less than or equal to dur.zero(), this function
         * behaves like {@code try_lock}. This function may continue blocking
         * after the timeout duration has elapsed due to scheduling or resource
         * contention delays.
         *
         * @remarks
         * It is usually not necessary to call @{code try_lock_for} directly.
         * use {@code std::unique_lock} to manage locking with timeouts.
         *
         * @param dur The minimum time the thread should block while waiting for
         *            the lock.
         * @return    true if the lock was aquired; otherwise, false.
         **/
        template <class Rep, class Period = std::ratio<1>>
        inline bool try_lock_for(const std::chrono::duration<Rep, Period>& dur) {
            using namespace std::chrono;

            if (try_lock()) {
                return true;
            }
            auto micros = duration_cast<microseconds>(dur);
            return try_lock_slow(micros);
        }

        /**
         * Attempts to aquire the lock. If another thread already owns the lock,
         * blocks execution of the current thread until the lock can be aquired
         * or until the specified timeout time is reached. A thread may aquire
         * the same {@code RecursiveLock} repeatedly. The lock will not be
         * released until a matching number of calls to {@code unlock} are made.
         *
         * @remarks
         * If {@p time} has already passed, this function behaves like
         * {@code try_lock}. This function may continue blocking after the
         * timeout time has been reached due to scheduling or resource
         * contention delays.
         *
         * @remarks
         * It is usually not necessary to call @{code try_lock_until} directly.
         * use {@code std::unique_lock} to manage locking with timeouts.
         *
         * @param time The earlierst time at which the thread should stop trying
         *             to aquire the lock.
         * @return     true if the lock was aquired; otherwise, false.
         **/
        template <class Clock, class Duration = typename Clock::duration>
        inline bool try_lock_until(const std::chrono::time_point<Clock, Duration>& time) {
            using namespace std::chrono;

            if (try_lock()) {
                return true;
            }

            auto now = Clock::now();
            if (now > time) {
                return false;
            }
            auto micros = duration_cast<microseconds>(time - now);
            return try_lock_slow(micros);
        }

        /**
         * If the lock has been aquired one more time than the number of calls
         * to {@code unlock}, releases the lock. Otherwise, reduces the internal
         * counter tracking the number of times the lock has been aquired by
         * one.
         *
         * @remarks
         * It is usually not necessary to call @{code unlock} directly. use
         * {@code std::unique_lock} and {@code std::lock_guard} to manage
         * locking.
         **/
        inline void unlock() {
            Expect(owner_id.load() == std::this_thread::get_id(),
                   "Calling unlock from a thread that does not own the lock");
            state_type value = state.load();
            if (value & count_mask > 1) {
                // The current thread has entered the lock multiple times. Since
                // we aren't touching the rest of the state and no other thread
                // is allowed to touch the count, we can just subtract 1. Since
                // the state lock is stored in the high bit, it won't be
                // affected unless another thread managed to change the count.
                state_type prev = state.fetch_sub(1, std::memory_order_release);
                Ensure(prev & count_mask > 1, "Modified count during unlock");
                return;
            }
            else {
                // We are about to release the mutex, which requires that we
                // check the state of the queue, which in turn requires us to
                // aquire the state lock, so off to slow poke land we go.
                unlock_slow();
            }
        }

    private:
        bool try_lock_slow(std::chrono::microseconds micros);

        void unlock_slow();

        void* wait_head;
        std::atomic<std::thread::id> owner_id;
        using state_type = std::uint16_t;
        state_type state_locked_bit = 0x8000;
        state_type count_mask = ~(state_locked_bit);
        std::atomic<state_type> state = 0x00;
    };
}

#endif
