// Copyright (c) 2017-2018 United States Government as represented by the
// Administrator of the National Aeronautics and Space Administration.
// All Rights Reserved.
#include "Lock.h"

namespace {
    struct ThreadData {
        bool waiting = false;
        std::mutex mutex;
        std::condition_variable condition;
        ThreadData* next;
        ThreadData* prev;
    };

    thread_local ThreadData thread_data;

    const std::memory_order mem_acq = std::memory_order_acquire;
    const std::memory_order mem_rel = std::memory_order_release;
    const std::memory_order mem_acq_rel = std::memory_order_acq_rel;

    /**
     * Trys to aquire a lock by atomically setting a bit that was not previously
     * set in an atomic variable. Continues trying in a loop until succesful,
     * yielding before each iteration.
     *
     * @param state    The atomic variable that holds the lock bit
     * @param lock_bit The bit field representing the lock within the state.
     * @return         The value just before the lock bit was succesfully set.
     **/
    template <typename T>
    T spin_aquire(std::atomic<T>& state, T lock_bit) {
        T value;
        for (;;) {
            std::this_thread::yield();
            value = state.load();

            if (!(value & lock_bit)) {
                // We should be able to get the lock, but we can fail either
                // spuriously or because we lost a race with someone else. If we
                // fail, just go back to the top and try again. If we succeed,
                // return the value prior to modification.
                T desired = value | lock_bit;
                if (state.compare_exchange_weak(value, desired, mem_acq)) {
                    return value;
                }
                else {
                    continue;
                }
            }
            else {
                // Someone else already has the lock; we need to wait for them
                // to finish.
                continue;
            }
        }
    }

    template <typename T>
    inline bool try_lock_once(std::atomic<T>& state, T value, T lock_bit) noexcept {
        if (!(value & lock_bit)) {
            // We should be able to get the lock, but we can fail either
            // spuriously or because we lost a race with someone else.
            T desired = value | lock_bit;
            if (state.compare_exchange_weak(value, desired, mem_acq)) {
                return true;
            }
        }

        // The lock was held before we loaded, or we failed to aquire it for
        // some reason. In any case, we failed to aquire it.
        return false;
    }

    /**
     * Remove the specified value from the queue pointed to by {@p head} if the
     * value is in the queue. Returns the new head, which may or may not be the
     * same as the original head.
     **/
    ThreadData* remove_from_queue(ThreadData* head, ThreadData* value) noexcept {
        if (!head) {
            // If there's no head, the queue is empty and trivially does not
            // contain value.
            value->next = nullptr;
            value->prev = nullptr;
            return nullptr;
        }
        if (head == value) {
            // value is at the head of the queue, so we need to replace the
            // head.
            ThreadData* new_head = value->next;
            if (new_head) {
                new_head->prev = value->prev;
            }
            value->next = nullptr;
            value->prev = nullptr;
            return new_head;
        }

        for (ThreadData* node = head->next; node != nullptr; node = node->next) {
            if (node != value) {
                continue;
            }
            // Note (JW): Since we skipped head, all nodes should have a
            // previous node, and that node should point back to us.
            Require(node->prev != nullptr, "No previous node in queue");
            Require(node->prev->next == node, "Previous node is queue tail");
            node->prev->next = node->next;
            if (node->next) {
                node->next->prev = node->prev;
            }
            value->next = nullptr;
            value->prev = nullptr;
            return head;
        }
    }

    /**
     * Insert the given value into the queue. If head is null, makes the value
     * the head of the queue. Returns the new head, which will be the original
     * head if it was non-null, or value if the original head was null.
     **/
    ThreadData* insert_into_queue(ThreadData* head, ThreadData* value) noexcept {
        if (!head) {
            // If there's no head, make value the head
            value->next = nullptr;
            value->prev = value;
            return value;
        }

        // Otherwise, put value at the end of the queue, which head points
        // backwards to.
        ThreadData* tail = head->prev;
        tail->next = value;
        value->prev = tail;
        value->next = nullptr;
        head->prev = value;
        return head;
    }

    ThreadData* pop_front(ThreadData*& head) noexcept {
        if (!head) {
            return nullptr;
        }
        ThreadData* result = head;
        head = result->next;
        if (head) {
            head->prev = result->prev;
        }
        result->next = nullptr;
        result->prev = nullptr;
        return result;
    }
}

namespace PCOE {
    void Lock::lock_slow() {
        // The fast path failed, either because of a spurious failure of the
        // cmp/ex, because someone else has the lock, or because there are
        // threads waiting for the lock. Whatever the reason, first spin several
        // more times to try to aquire the lock. If that fails, add our thread
        // to the wait queue and wait there until the lock is released or we
        // time out.
        unsigned spin_count = 40;
        for (;;) {
            // Each iteration of the loop will execute based on the same value.
            // If the value changes, we stop and start over from the top.
            state_type value = state.load();

            // Always try to aquire the lock first.
            if (try_lock_once(state, value, locked_bit)) {
                return;
            }
            if (spin_count > 0) {
                // Still spinning; don't try to wait on the thread mutex yet.
                --spin_count;
                std::this_thread::yield();
                continue;
            }

            // We've given up on spinning. Try to aquire the queue lock.
            Require(value & locked_bit, "Main lock not held while attempting to lock queue");
            if (!try_lock_once(state, value, queue_locked_bit)) {
                std::this_thread::yield();
                continue;
            }

            // By now, we've given up spinning and we've aquired the queue lock,
            // so we can freely manipulate the wait queue. Get the queue and put
            // our thread in it, then wait for our thread to be woken up.
            ThreadData* head = reinterpret_cast<ThreadData*>(value & queue_mask);
            ThreadData* new_head = insert_into_queue(head, &thread_data);
            if (head != new_head) {
                // Head was modified. Need to store the result.
                state_type new_state = reinterpret_cast<state_type>(&head);
                Require(new_state & queue_mask == new_state, "ThreadData pointer is not aligned");
                // We own the queue lock, so there's no nead for a cmp/ex here.
                // We can safely store the new head. If we've gotten this far,
                // both locks must be taken. We want to set the queue head and
                // preserve both lock bits.
                state_type new_state = new_state | queue_locked_bit | locked_bit;
                state.store(new_state, std::memory_order_relaxed);
            }

            // We've added ourselves to the wait queue. Now release the queue
            // lock and wait to be woken up. Each time we're woken up, check to
            // see if we're still waiting. If so, the wakeup was spurious and we
            // need to wait again.
            std::unique_lock<std::mutex> lock(thread_data.mutex);
            thread_data.waiting = true;
            state.fetch_sub(queue_locked_bit, mem_rel);
            while (thread_data.waiting) {
                thread_data.condition.wait(lock);
            }
        }
    }

    void Lock::unlock_slow() {
        // The fast path failed, either spuriously or because there are threads
        // waiting. Since someone is calling unlock, we should always see the
        // locked bit set. We need to aquire the queue lock and see whether
        // anyone is really waiting.
        state_type value = spin_aquire(state, queue_locked_bit);
        Require(value & locked_bit, "Calling unlock on unlocked mutex");

        // Now we have the queue lock and the main lock is held. If anyone is
        // waiting, we want to pop the head of the queue and signal the waiting
        // thread. We then reset the head of the queue (either to the next
        // waiting thread or to nothing) and release both locks.
        ThreadData* head = reinterpret_cast<ThreadData*>(value & queue_mask);
        ThreadData* result = pop_front(head);
        if (result) {
            // Popped a value off the queue. Need to store the new head and
            // notify the popped value.
            state_type new_head = reinterpret_cast<state_type>(head);
            Require(new_head & queue_mask == new_head, "New head has unaligned address");

            // The waiting thread has to release the queue lock before it
            // finally calls wait. To make sure we notfiy it after it starts
            // waiting, we need to briefly lock its mutex. Since the lock
            // operation locks the thread mutex while still holding the queue
            // lock just like us, there is no risk of a deadlock due to
            // ordering.
            std::unique_lock<std::mutex> lock(head->mutex);
            result->waiting = false;
            lock.unlock();
            result->condition.notify_one();

            // Set the new head and release the main and queue locks
            // simultaneously.
            state.store(new_head, mem_rel);
        }
        else {
            // The fast path must have failed spuriously, becasue no one is
            // actually waiting. Since no one is waiting, we can release both
            // the queue lock and the main lock, putting the lock back onto the
            // fast path.
            state.store(0, mem_rel);
        }
    }

    bool TimedLock::try_lock_slow(std::chrono::microseconds millis) {
        using namespace std::chrono;
        auto timeout_at = steady_clock::now() + millis;

        // The fast path failed, either because of a spurious failure of the
        // cmp/ex, because someone else has the lock, or because there are
        // threads waiting for the lock. Whatever the reason, first spin several
        // more times to try to aquire the lock. If that fails, add our thread
        // to the wait queue and wait there until the lock is released or we
        // time out.
        unsigned spin_count = 40;
        while (timeout_at >= steady_clock::now()) {
            // Each iteration of the loop will execute based on the same value.
            // If the value changes, we stop and start over from the top.
            state_type value = state.load();

            // Always try to aquire the lock first.
            if (try_lock_once(state, value, locked_bit)) {
                return;
            }
            if (spin_count > 0) {
                // Still spinning; don't try to wait on the thread mutex yet.
                --spin_count;
                std::this_thread::yield();
                continue;
            }

            // We've given up on spinning. Try to aquire the queue lock.
            Require(value & locked_bit, "Main lock not held while attempting to lock queue");
            if (!try_lock_once(state, value, queue_locked_bit)) {
                std::this_thread::yield();
                continue;
            }

            // By now, we've given up spinning and we've aquired the queue lock,
            // so we can freely manipulate the wait queue. Get the queue and put
            // our thread in it, then wait for our thread to be woken up.
            ThreadData* head = reinterpret_cast<ThreadData*>(value & queue_mask);
            ThreadData* new_head = insert_into_queue(head, &thread_data);
            if (head != new_head) {
                // Head was modified. Need to store the result.
                state_type new_state = reinterpret_cast<state_type>(&head);
                Require(new_state & queue_mask == new_state, "ThreadData pointer is not aligned");
                // We own the queue lock, so there's no nead for a cmp/ex here.
                // We can safely store the new head. If we've gotten this far,
                // both locks must be taken. We want to set the queue head and
                // preserve both lock bits.
                state_type new_state = new_state | queue_locked_bit | locked_bit;
                state.store(new_state, std::memory_order_relaxed);
            }

            // We've added ourselves to the wait queue. Now release the queue
            // lock and wait to be woken up. Each time we're woken up, check to
            // see if we're still waiting. If so, the wakeup was spurious and we
            // need to wait again.
            std::unique_lock<std::mutex> lock(thread_data.mutex);
            thread_data.waiting = true;
            state.fetch_sub(queue_locked_bit, mem_rel);
            std::cv_status status = std::cv_status::no_timeout;
            while (thread_data.waiting && status != std::cv_status::timeout) {
                status = thread_data.condition.wait_until(lock, timeout_at);
            }

            // We've been woken up. Remove ourselves from the queue and try
            // again. First, spin to aquire the queue lock. Once we have it,
            // Find and remove ourselves from the queue. Finally, loop again. If
            // we timed out, we will exit early. If it was a spurious wakeup or
            // another thread got to the lock before us, we will end up at the
            // end of the queue again, but most likely we will find the lock
            // available and succeed in aquiring it.
            bool aquired_lock = false;
            for (;;) {
                value = state.load();

                // We will try to get the lock when we loop again anyway, but we
                // want to maintain the invariant that the queue lock is only
                // held when the main lock is held. If no one else is holding
                // the main lock, we should go ahead and take it. That's what
                // we're here for anyway.
                if (!(value & locked_bit)) {
                    state_type desired = value | locked_bit;
                    if (state.compare_exchange_weak(value, desired, mem_acq)) {
                        aquired_lock = true;
                        value = desired;
                    }
                    else {
                        continue;
                    }
                }

                Require(value & locked_bit, "Main lock not held while trying to lock queue");
                if (!try_lock_once(state, value, queue_locked_bit)) {
                    std::this_thread::yield();
                    continue;
                }

                // Now we have the queue lock and someone has the main lock.
                // Since we were waiting, we know that the head of the queue is
                // non-null, forcing unlocks into the slow path, where they have
                // to aquire the queue lock to release the main lock. Therefore,
                // we know that both locks are held and that we are the only
                // thread that can modify the state.
                ThreadData* head = reinterpret_cast<ThreadData*>(value & queue_mask);
                ThreadData* new_head = remove_from_queue(head, &thread_data);
                if (head != new_head) {
                    // Our thread data was at the head of the queue. Need to
                    // store the new head. We simultaneously release
                    // the queue lock, since we're done modifying the queue.
                    state_type new_state = reinterpret_cast<state_type>(new_head);
                    Require(new_state & queue_mask == new_state, "New head not aligned");
                    state.store(new_state | locked_bit, mem_rel);
                }
                else {
                    // Head didn't change, so we just need to release the queue
                    // lock.
                    state.fetch_sub(queue_locked_bit, mem_rel);
                }

                if (aquired_lock) {
                    // Lucky us, we got the lock here instead of looping again.
                    // We're done and we have the lock.
                    return true;
                }
                else {
                    // Whatever the reason, we don't have the lock, so we head
                    // back to the top of the main loop, where we will either
                    // time out or try again.
                    break;
                }
            }
        }
        return false;
    }

    void TimedLock::unlock_slow() {
        // The fast path failed, either spuriously or because there are threads
        // waiting. Since someone is calling unlock, we should always see the
        // locked bit set. We need to aquire the queue lock and see whether
        // anyone is really waiting.
        state_type value = spin_aquire(state, queue_locked_bit);
        Require(value & locked_bit, "Calling unlock on unlocked mutex");

        // Now we hold the queue lock and the main lock is held. We want to get
        // the head of the queue and signal them, causing them to wake up and
        // try to aquire the lock for themselves. We don't modify the queue
        // since waiting threads have to take care of that themselves anyway in
        // the case where they time out. We just wake up the thread and let it
        // figure things out.
        ThreadData* head = reinterpret_cast<ThreadData*>(value & queue_mask);
        if (head) {
            // The waiting thread has to release the queue lock before it
            // finally calls wait. To make sure we notfiy it after it starts
            // waiting, we need to briefly lock its mutex. Since the lock
            // operation locks the thread mutex while still holding the queue
            // mutex just like us, there is no risk of a deadlock due to
            // ordering. Once we've changed the waiting thread's status, we drop
            // all of the locks before notifying the waiting thread so that it
            // doesn't have to wait on us after we wake it up.
            std::unique_lock<std::mutex> lock(head->mutex);
            head->waiting = false;
            lock.unlock();
            state.fetch_sub(locked_bit | queue_locked_bit, mem_rel);
            head->condition.notify_one();
        }
        else {
            // The fast path must have failed spuriously, becasue no one is
            // actually waiting. Since no one is waiting, we can just clear the
            // state to release both the main and queue locks.
            state.store(0, mem_rel);
        }
    }

    bool RecursiveLock::try_lock_slow(std::chrono::microseconds millis) {
        using namespace std::chrono;
        auto timeout_at = steady_clock::now() + millis;

        // The fast path failed, either because of a spurious failure of the
        // cmp/ex, because someone else has the lock, or because there are
        // threads waiting for the lock. Whatever the reason, first spin several
        // more times to try to aquire the lock. If that fails, add our thread
        // to the wait queue and wait there until the lock is released or we
        // time out.
        unsigned spin_count = 40;
        while (timeout_at >= steady_clock::now()) {
            // Each iteration of the loop will execute based on the same value.
            // If the value changes, we stop and start over from the top.
            state_type value = state.load();

            if (!value) {
                // No one owns the thread, so we can try to aquire the lock
                // immediately. We try to aquire the main lock with a count of 1
                // and the state lock simultaneously. We modify the owner id
                // while holding the state lock, then release the state lock
                // before returning. If we fail for some reason, go back to the
                // top and try again.
                state_type desired = 0x0001 | state_locked_bit;
                if (state.compare_exchange_weak(value, desired, mem_acq)) {
                    owner_id.store(std::this_thread::get_id());
                    state.fetch_sub(state_locked_bit, mem_rel);
                    return true;
                }
                else {
                    continue;
                }
            }
            if (spin_count > 0) {
                // Still spinning; don't try to wait on the thread mutex yet.
                --spin_count;
                std::this_thread::yield();
                continue;
            }

            // We've given up on spinning. Try to aquire the state lock.
            if (!try_lock_once(state, value, state_locked_bit)) {
                std::this_thread::yield();
                continue;
            }

            // By now, we've given up spinning and we've got the state lock, so
            // we can freely manipulate the wait queue. Get the queue and put
            // our thread in it, then wait for our thread to be woken up.
            ThreadData* head = reinterpret_cast<ThreadData*>(wait_head);
            wait_head = reinterpret_cast<void*>(insert_into_queue(head, &thread_data));

            // We've added ourselves to the wait queue. Now release the state
            // lock and wait to be woken up. Each time we're woken up, check to
            // see if we're still waiting. If so, the wakeup was spurious and we
            // need to wait again.
            std::unique_lock<std::mutex> lock(thread_data.mutex);
            thread_data.waiting = true;
            state.fetch_sub(state_locked_bit, mem_rel);
            std::cv_status status = std::cv_status::no_timeout;
            while (thread_data.waiting && status != std::cv_status::timeout) {
                status = thread_data.condition.wait_until(lock, timeout_at);
            }

            // We've been woken up. Remove ourselves from the queue and try
            // again. First, spin to aquire the state lock. Once we have it,
            // Find and remove ourselves from the queue. Finally, loop again. If
            // we timed out, we will exit early. If it was a spurious wakeup or
            // another thread got to the lock before us, we will end up at the
            // end of the queue again, but most likely we will find the lock
            // available and succeed in aquiring it.
            bool aquired_lock = false;
            for (;;) {
                value = state.load();

                // We will try to get the lock when we loop again anyway, but we
                // want to maintain the invariant that the queue lock is only
                // held when the main lock is held. If no one else is holding
                // the main lock, we should go ahead and take it. That's what
                // we're here for anyway.
                if (!value) {
                    state_type desired = 0x0001;
                    if (state.compare_exchange_weak(value, desired, mem_acq)) {
                        aquired_lock = true;
                        value = desired;
                    }
                    else {
                        continue;
                    }
                }

                Require(value & count_mask, "Main lock not held while trying to lock queue");
                if (!try_lock_once(state, value, state_locked_bit)) {
                    std::this_thread::yield();
                    continue;
                }

                // Now we have the state lock and someone has the main lock.
                // Since even the fast unlock path has to take the state lock to
                // reset the owner id, we know that no one else can touch the
                // queue. Therefore, we can safely remove ourselves from the
                // queue and carry on.
                ThreadData* head = reinterpret_cast<ThreadData*>(wait_head);
                wait_head = reinterpret_cast<void*>(remove_from_queue(head, &thread_data));

                if (aquired_lock) {
                    // Lucky us, we got the lock here instead of looping again.
                    // We just need to update the owner id, release the state
                    // lock, and we're done.
                    owner_id = std::this_thread::get_id();
                    state.fetch_sub(state_locked_bit, mem_rel);
                    return true;
                }
                else {
                    // Whatever the reason, we don't have the main lock, so we
                    // release the state lock and head back to the top of the
                    // main loop, where we will either time out or try again.
                    state.fetch_sub(state_locked_bit, mem_rel);
                    break;
                }
            }
        }
        return false;
    }

    void RecursiveLock::unlock_slow() {
        // The fast path failed, either spuriously or because there are threads
        // waiting. Since someone is calling unlock, we should always see the
        // locked bit set. We need to aquire the state lock and see whether
        // anyone is really waiting.
        state_type value = spin_aquire(state, state_locked_bit);
        Require(value & count_mask, "Calling unlock on unlocked mutex");
        Require(value & count_mask == 1, "Calling slow path on recursively locked mutex");

        // We're about to release the main lock, so reset the owner id.
        owner_id.store(std::thread::id());

        // Now we hold the state lock and the main lock is held. We want to get
        // the head of the queue and signal them, causing them to wake up and
        // try to aquire the lock for themselves. We don't modify the queue
        // since waiting threads have to take care of that themselves anyway in
        // the case where they time out. We just wake up the thread and let it
        // figure things out.
        ThreadData* head = reinterpret_cast<ThreadData*>(wait_head);
        if (head) {
            // The waiting thread has to release the state lock before it
            // finally calls wait. To make sure we notfiy it after it starts
            // waiting, we need to briefly lock its mutex. Since the lock
            // operation locks the thread mutex while still holding the state
            // lock just like us, there is no risk of a deadlock due to
            // ordering. Once we've changed the waiting thread's status, we drop
            // all of the locks before notifying the waiting thread so that it
            // doesn't have to wait on us after we wake it up.
            std::unique_lock<std::mutex> lock(head->mutex);
            head->waiting = false;
            lock.unlock();
            state.store(0, mem_rel);
            head->condition.notify_one();
        }
        else {
            // The fast path must have failed spuriously, becasue no one is
            // actually waiting. Since no one is waiting, we can just clear the
            // state to release both the main and state locks.
            state.store(0, mem_rel);
        }
    }
}
