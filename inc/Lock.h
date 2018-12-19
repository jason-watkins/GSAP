// Copyright (c) 2017-2018 United States Government as represented by the
// Administrator of the National Aeronautics and Space Administration.
// All Rights Reserved.
#ifndef PCOE_LOCK_H
#define PCOE_LOCK_H
#include <atomic>
#include <mutex>
#include <thread>

#include "Contracts.h"

namespace PCOE {
    /**
     * Provides a locking primitive that avoids the context switches of an
     * OS-provided mutex when possible.
     *
     * @remarks
     * In the case where a lock is uncontested, locks and unlocks amount to a
     * single compare/exchange operation each. If a thread fails to immediately
     * aquire the lock, it will spin several times waiting for the lock to
     * become available. Finally, if that fails, the thread will wait on an OS
     * mutex.
     *
     * @remarks
     * Since any given thread can only be waiting on a single lock at any given
     * time, the classes provided by this module allocate one OS mutex per
     * thread, rather than one per lock. This makes the locks themselves
     * relatively small (a single pointer), while introducing minimal overhead
     * per already relatively expensive thread.
     *
     * @remarks
     * This class was inspired by https://webkit.org/blog/6161/locking-in-webkit/
     *
     * @author Jason Watkins
     **/
    class Lock {
    public:
        inline void lock() {
            state_type expected = 0;
            state_type desired = locked_bit;
            if (state.compare_exchange_strong(expected, desired, std::memory_order_acquire)) {
                return;
            }

            lock_slow();
        }

        inline bool try_lock() {
            state_type value = state.load();
            state_type expected = value & ~locked_bit;
            state_type desired = value | locked_bit;
            return state.compare_exchange_strong(expected, desired, std::memory_order_acquire);
        }

        void unlock() {
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
        state_type locked_bit = 0x01;
        state_type queue_locked_bit = 0x02;
        state_type queue_mask = ~(locked_bit | queue_locked_bit);
        std::atomic<state_type> state = 0x00;
    };

    /**
     * Provides a locking primitive that avoids the context switches of an
     * OS-provided mutex when possible. The {@code TimedLock} expands on the
     * {@code Lock} class by providing functions for locking with a timeout.
     *
     * @remarks
     * In the case where a lock is uncontested, locks and unlocks amount to a
     * single compare/exchange operation each. If a thread fails to immediately
     * aquire the lock, it will spin several times waiting for the lock to
     * become available. Finally, if that fails, the thread will wait on an OS
     * mutex.
     *
     * @remarks
     * Since any given thread can only be waiting on a single lock at any given
     * time, the classes provided by this module allocate one OS mutex per
     * thread, rather than one per lock. This makes the locks themselves
     * relatively small (a single pointer), while introducing minimal overhead
     * per already relatively expensive thread.
     *
     * @remarks
     * This class was inspired by https://webkit.org/blog/6161/locking-in-webkit/
     *
     * @author Jason Watkins
     **/
    class TimedLock {
    public:
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

        inline bool try_lock() {
            state_type value = state.load();
            state_type expected = value & ~locked_bit;
            state_type desired = value | locked_bit;
            return state.compare_exchange_strong(expected, desired, std::memory_order_acquire);
        }

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

        template <class Rep, class Period = std::ratio<1>>
        inline bool try_lock_for(const std::chrono::duration<Rep, Period>& dur) {
            using namespace std::chrono;

            if (try_lock()) {
                return true;
            }
            auto micros = duration_cast<microseconds>(dur);
            return try_lock_slow(micros);
        }

        void unlock() {
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
     * Provides a locking primitive that avoids the context switches of an
     * OS-provided mutex when possible. The {@code TimedLock} expands on the
     * {@code Lock} class by allowing the same thread to call lock multiple
     * times. The aditional overhead of timing is judged to be minimal compared
     * to the overhead of recursion, so all recursive locks also provide the
     * option of locking with a timeout.
     *
     * @remarks
     * A recursive lock has to do substantially more bookkeeping that a
     * non-recursive lock. In the case where a thread already owns the lock and
     * is entering or leaving recursively, locks and unlocks amount to an atomic
     * load and an atomic add each. In the case where a thread is aquiring an
     * uncontested lock for the first time, the thread must perform several
     * atomic operations. All other cases will fall back to the "slow" path,
     * where the thread will spin several times waiting for the lock to become
     * available, and finally fall back to waiting on an OS mutex.
     *
     * @remarks
     * Since any given thread can only be waiting on a single lock at any given
     * time, the classes provided by this module allocate one OS mutex per
     * thread, rather than one per lock. This makes the locks themselves
     * relatively small (a pointer, a thread id and a counter for a recursive
     * lock), while introducing minimal overhead per already relatively
     * expensive thread.
     *
     * @remarks
     * This class was inspired by
     * https://webkit.org/blog/6161/locking-in-webkit/
     *
     * @author Jason Watkins
     **/
    class RecursiveLock {
    public:
        inline void lock() {
            bool locked = try_lock();
            while (!locked) {
                locked = try_lock_slow(std::chrono::microseconds::max());
            }
        }

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

        template <class Rep, class Period = std::ratio<1>>
        inline bool try_lock_for(const std::chrono::duration<Rep, Period>& dur) {
            using namespace std::chrono;

            if (try_lock()) {
                return true;
            }
            auto micros = duration_cast<microseconds>(dur);
            return try_lock_slow(micros);
        }

        void unlock() {
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
