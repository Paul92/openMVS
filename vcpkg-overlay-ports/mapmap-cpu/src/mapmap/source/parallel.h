#ifndef MAPMAP_SOURCE_PARALLEL_H_
#define MAPMAP_SOURCE_PARALLEL_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <iterator>
#include <mutex>
#include <queue>
#include <thread>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tbb {

struct split {};
struct pre_scan_tag {};
struct final_scan_tag {};

template <typename Index>
class blocked_range {
public:
    using const_iterator = Index;

    blocked_range(Index begin, Index end, Index grainsize = static_cast<Index>(1))
        : begin_(begin), end_(end), grainsize_(grainsize > 0 ? grainsize : static_cast<Index>(1)) {}

    Index begin() const { return begin_; }
    Index end() const { return end_; }
    Index grainsize() const { return grainsize_; }
    bool empty() const { return begin_ >= end_; }

private:
    Index begin_;
    Index end_;
    Index grainsize_;
};

namespace detail {

template <typename Index>
inline int compute_chunks(const blocked_range<Index>& range) {
    const auto n = static_cast<long long>(range.end() - range.begin());
    if (n <= 0) {
        return 0;
    }
#ifdef _OPENMP
    const int threads = omp_get_max_threads();
#else
    const int threads = 1;
#endif
    const auto grain = static_cast<long long>(range.grainsize());
    const auto by_grain = static_cast<int>((n + grain - 1) / grain);
    const int target = std::max(1, threads * 4);
    return std::max(1, std::min(by_grain, target));
}

template <typename Index>
inline blocked_range<Index> make_chunk(const blocked_range<Index>& range, int chunk, int chunks) {
    const auto begin = range.begin();
    const auto n = static_cast<long long>(range.end() - begin);
    const auto c0 = static_cast<long long>(chunk);
    const auto c1 = static_cast<long long>(chunk + 1);
    const auto k = static_cast<long long>(chunks);
    const Index cb = static_cast<Index>(begin + static_cast<Index>((n * c0) / k));
    const Index ce = static_cast<Index>(begin + static_cast<Index>((n * c1) / k));
    return blocked_range<Index>(cb, ce, range.grainsize());
}

} // namespace detail

template <typename Index, typename Func>
inline void parallel_for(const blocked_range<Index>& range, const Func& func) {
    const int chunks = detail::compute_chunks(range);
    if (chunks <= 0) {
        return;
    }
    if (chunks == 1) {
        func(range);
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < chunks; ++c) {
        const auto chunk = detail::make_chunk(range, c, chunks);
        if (!chunk.empty()) {
            func(chunk);
        }
    }
}

template <typename Index, typename Value, typename Func, typename Reduce>
inline Value parallel_reduce(
    const blocked_range<Index>& range,
    Value init,
    const Func& func,
    const Reduce& reduce) {
    const int chunks = detail::compute_chunks(range);
    if (chunks <= 0) {
        return init;
    }
    if (chunks == 1) {
        return func(range, init);
    }

    std::vector<Value> partial(static_cast<size_t>(chunks), init);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < chunks; ++c) {
        const auto chunk = detail::make_chunk(range, c, chunks);
        partial[static_cast<size_t>(c)] = func(chunk, init);
    }

    Value out = init;
    for (int c = 0; c < chunks; ++c) {
        out = reduce(out, partial[static_cast<size_t>(c)]);
    }
    return out;
}

template <typename Index, typename Body>
inline void parallel_scan(const blocked_range<Index>& range, Body& body) {
    const int chunks = detail::compute_chunks(range);
    if (chunks <= 0) {
        return;
    }
    if (chunks == 1) {
        body(range, final_scan_tag{});
        return;
    }

    std::vector<Body> partial;
    partial.reserve(static_cast<size_t>(chunks));
    for (int c = 0; c < chunks; ++c) {
        partial.emplace_back(body, split{});
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < chunks; ++c) {
        const auto chunk = detail::make_chunk(range, c, chunks);
        if (!chunk.empty()) {
            partial[static_cast<size_t>(c)](chunk, pre_scan_tag{});
        }
    }

    for (int c = 1; c < chunks; ++c) {
        partial[static_cast<size_t>(c)].reverse_join(partial[static_cast<size_t>(c - 1)]);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int c = 0; c < chunks; ++c) {
        const auto chunk = detail::make_chunk(range, c, chunks);
        if (chunk.empty()) {
            continue;
        }
        Body worker(body, split{});
        if (c > 0) {
            worker.assign(partial[static_cast<size_t>(c - 1)]);
        }
        worker(chunk, final_scan_tag{});
    }

    body.assign(partial.back());
}

template <typename T>
class feeder {
public:
    feeder(std::deque<T>* queue, std::mutex* mutex)
        : queue_(queue), mutex_(mutex) {}

    void add(const T& value) {
        std::lock_guard<std::mutex> lock(*mutex_);
        queue_->push_back(value);
    }

    void add(T&& value) {
        std::lock_guard<std::mutex> lock(*mutex_);
        queue_->push_back(std::move(value));
    }

private:
    std::deque<T>* queue_;
    std::mutex* mutex_;
};

template <typename Iterator, typename Func>
inline void parallel_for_each(Iterator begin, Iterator end, Func func) {
    using Value = typename std::iterator_traits<Iterator>::value_type;

    constexpr bool kUsesFeeder =
        std::is_invocable_v<Func, const Value&, feeder<Value>&> ||
        std::is_invocable_v<Func, Value, feeder<Value>&>;

    if constexpr (!kUsesFeeder) {
        std::vector<Value> items(begin, end);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(items.size()); ++i) {
            const Value& item = items[static_cast<std::size_t>(i)];
            if constexpr (std::is_invocable_v<Func, const Value&>) {
                func(item);
            } else {
                func(item);
            }
        }
        return;
    }

    std::deque<Value> queue;
    for (auto it = begin; it != end; ++it) {
        queue.push_back(*it);
    }
    if (queue.empty()) {
        return;
    }

    std::mutex mutex;
    std::size_t active = 0;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        feeder<Value> fd(&queue, &mutex);

        for (;;) {
            Value item{};
            bool have_item = false;

            {
                std::unique_lock<std::mutex> lock(mutex);
                if (!queue.empty()) {
                    item = std::move(queue.front());
                    queue.pop_front();
                    ++active;
                    have_item = true;
                } else if (active == 0) {
                    break;
                }
            }

            if (!have_item) {
                std::this_thread::yield();
                continue;
            }

            if constexpr (std::is_invocable_v<Func, const Value&, feeder<Value>&>) {
                func(item, fd);
            } else if constexpr (std::is_invocable_v<Func, Value, feeder<Value>&>) {
                func(item, fd);
            } else if constexpr (std::is_invocable_v<Func, const Value&>) {
                func(item);
            } else {
                func(item);
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                --active;
            }
        }
    }
}

template <typename T, typename Allocator = std::allocator<T>>
class concurrent_vector {
public:
    using value_type = T;
    using allocator_type = Allocator;
    using size_type = typename std::vector<T, Allocator>::size_type;
    using iterator = typename std::vector<T, Allocator>::iterator;
    using const_iterator = typename std::vector<T, Allocator>::const_iterator;

    concurrent_vector() = default;

    explicit concurrent_vector(size_type n)
        : data_(n) {}

    template <typename InputIt>
    concurrent_vector(InputIt first, InputIt last)
        : data_(first, last) {}

    size_type size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.empty();
    }

    void reserve(size_type n) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.reserve(n);
    }

    void resize(size_type n) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.resize(n);
    }

    void resize(size_type n, const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.resize(n, value);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
    }

    template <typename InputIt>
    void assign(InputIt first, InputIt last) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.assign(first, last);
    }

    void assign(size_type count, const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.assign(count, value);
    }

    void push_back(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.push_back(value);
    }

    void push_back(T&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.push_back(std::move(value));
    }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.emplace_back(std::forward<Args>(args)...);
    }

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }
    const_iterator cbegin() const { return data_.cbegin(); }
    const_iterator cend() const { return data_.cend(); }

    T& front() { return data_.front(); }
    const T& front() const { return data_.front(); }
    T& back() { return data_.back(); }
    const T& back() const { return data_.back(); }

    T& operator[](size_type idx) { return data_[idx]; }
    const T& operator[](size_type idx) const { return data_[idx]; }

private:
    std::vector<T, Allocator> data_;
    mutable std::mutex mutex_;
};

template <typename T>
class concurrent_queue {
public:
    concurrent_queue() = default;

    void push(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(value);
    }

    void push(T&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(std::move(value));
    }

    bool try_pop(T& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        out = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
};

template <typename Key,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>,
          typename Allocator = std::allocator<Key>>
class concurrent_unordered_set {
public:
    using value_type = Key;
    using size_type = typename std::unordered_set<Key, Hash, KeyEqual, Allocator>::size_type;
    using iterator = typename std::unordered_set<Key, Hash, KeyEqual, Allocator>::iterator;
    using const_iterator = typename std::unordered_set<Key, Hash, KeyEqual, Allocator>::const_iterator;

    explicit concurrent_unordered_set(size_type bucket_count = 0)
        : data_(bucket_count) {}

    std::pair<iterator, bool> insert(const Key& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.insert(value);
    }

    std::pair<iterator, bool> insert(Key&& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.insert(std::move(value));
    }

    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

private:
    std::unordered_set<Key, Hash, KeyEqual, Allocator> data_;
    mutable std::mutex mutex_;
};

class task_group {
public:
    task_group() = default;

    template <typename Func>
    void run(Func&& func) {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.emplace_back(std::forward<Func>(func));
    }

    void wait() {
        std::vector<std::function<void()>> tasks;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks.swap(tasks_);
        }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(tasks.size()); ++i) {
            tasks[static_cast<std::size_t>(i)]();
        }
    }

private:
    std::vector<std::function<void()>> tasks_;
    std::mutex mutex_;
};

class task_arena {
public:
    static constexpr int automatic = -1;

    explicit task_arena(int max_concurrency = automatic)
        : max_concurrency_(max_concurrency > 0 ? max_concurrency : default_concurrency()) {
#ifdef _OPENMP
        if (max_concurrency > 0) {
            omp_set_num_threads(max_concurrency);
        }
#endif
    }

    int max_concurrency() const {
        return max_concurrency_;
    }

    template <typename Func>
    auto execute(Func&& func) -> decltype(func()) {
        return func();
    }

private:
    static int default_concurrency() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return static_cast<int>(std::thread::hardware_concurrency());
#endif
    }

    int max_concurrency_;
};

class tick_count {
public:
    class interval_t {
    public:
        explicit interval_t(double seconds)
            : seconds_(seconds) {}

        double seconds() const { return seconds_; }

    private:
        double seconds_;
    };

    tick_count()
        : time_(clock_t::now()) {}

    static tick_count now() {
        return tick_count(clock_t::now());
    }

    interval_t operator-(const tick_count& other) const {
        return interval_t(std::chrono::duration<double>(time_ - other.time_).count());
    }

private:
    using clock_t = std::chrono::steady_clock;

    explicit tick_count(clock_t::time_point tp)
        : time_(tp) {}

    clock_t::time_point time_;
};

} // namespace tbb

#endif // MAPMAP_SOURCE_PARALLEL_H_
