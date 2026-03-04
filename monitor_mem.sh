#!/usr/bin/env bash
# Memory monitor for MemoryTimer. Polls target process and direct children
# at a fixed interval, printing "kb, %" per sample. Killed by MemoryTimer.
# Usage: monitor_mem.sh -p PID [-i INTERVAL_SECONDS]

DEFAULT_INTERVAL=0.01

TARGET_PID=""
INTERVAL="$DEFAULT_INTERVAL"

while getopts ":p:i:" opt; do
    case ${opt} in
        p ) TARGET_PID=$OPTARG ;;
        i ) INTERVAL=$OPTARG ;;
        \? ) echo "Invalid option: $OPTARG" 1>&2; exit 1 ;;
        : ) echo "Option $OPTARG requires an argument" 1>&2; exit 1 ;;
    esac
done

trap 'exit 0' SIGINT SIGTERM

TOTAL_MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)

# Page size in KB — converts statm page counts to KB.
PAGE_SIZE_KB=$(( $(getconf PAGE_SIZE) / 1024 ))

# Exclude this monitor script from the children list (it is itself a direct
# child of the Python process via subprocess.Popen).
SELF_PID=$$

# Baseline system-wide shmem. multiprocessing.heap.Arena creates /dev/shm
# files; workers fault pages in but main's page table stays empty until
# first access, so those pages don't appear in any PSS. The delta captures
# them.
SHMEM_BASELINE=$(awk '/^Shmem:/ {print $2; exit}' /proc/meminfo)

while true; do
    if [ ! -d "/proc/$TARGET_PID" ]; then
        exit 0
    fi

    # Read direct children of main thread (TID==PID) only. Python workers are
    # fork()ed from TID==PID; OpenMP/BLAS threads (CLONE_THREAD) cannot appear
    # here. No subprocess forks — read builtin only.
    # /proc files often lack a trailing newline; read returns 1 on EOF-without-
    # newline even after reading content, so don't use its exit code.
    _ch_line=""
    read -r _ch_line \
        < "/proc/$TARGET_PID/task/$TARGET_PID/children" 2>/dev/null || :
    CHILDREN=""
    for _cpid in $_ch_line; do
        [ "$_cpid" != "$SELF_PID" ] && \
            CHILDREN="${CHILDREN:+$CHILDREN }$_cpid"
    done

    if [ -z "$CHILDREN" ]; then
        # No workers: use statm (atomic RSS counters, no kernel lock).
        # smaps_rollup acquires mmap_read_lock() and contends with BLAS
        # mmap/munmap — inflates PCA by 2×. RSS = PSS for private anonymous
        # pages (all memory during PCA/kNN/cluster/embed).
        TOTAL_PSS=0
        if read -r _ _rss _ _ _ _ _ \
                < "/proc/$TARGET_PID/statm" 2>/dev/null; then
            TOTAL_PSS=$(( _rss * PAGE_SIZE_KB ))
        fi
    else
        # Workers present (multi-threaded loading): use smaps_rollup PSS +
        # Shmem delta to capture /dev/shm Arena pages workers wrote before
        # main has touched them.

        # Main process PSS and its shmem component in one awk pass.
        MAIN_PSS=0; MAIN_PSS_SHMEM=0
        if [ -f "/proc/$TARGET_PID/smaps_rollup" ]; then
            read -r MAIN_PSS MAIN_PSS_SHMEM < <(awk '
                /^Pss:/       { pss   = $2 }
                /^Pss_Shmem:/ { shmem = $2 }
                END           { print pss+0, shmem+0 }
            ' "/proc/$TARGET_PID/smaps_rollup" 2>/dev/null)
        fi

        # PSS distributes shared pages proportionally — summing across the
        # process tree counts shared memory exactly once.
        CHILDREN_PSS=0
        for pid in $CHILDREN; do
            if [ -f "/proc/$pid/smaps_rollup" ]; then
                PSS=$(awk '/^Pss:/ {print $2; exit}' \
                    "/proc/$pid/smaps_rollup" 2>/dev/null)
                [ -n "$PSS" ] && CHILDREN_PSS=$((CHILDREN_PSS + PSS))
            fi
        done
        TOTAL_PSS=$((MAIN_PSS + CHILDREN_PSS))

        # Shmem candidate: new global shmem minus what main already sees via
        # Pss_Shmem (avoids double-counting after main touches the pages).
        SHMEM_CURRENT=$(awk '/^Shmem:/ {print $2; exit}' /proc/meminfo)
        SHMEM_DELTA=$(( SHMEM_CURRENT - SHMEM_BASELINE ))
        [ "$SHMEM_DELTA" -lt 0 ] && SHMEM_DELTA=0
        UNCOUNTED_SHMEM=$(( SHMEM_DELTA - MAIN_PSS_SHMEM ))
        [ "$UNCOUNTED_SHMEM" -lt 0 ] && UNCOUNTED_SHMEM=0
        SHMEM_CANDIDATE=$(( MAIN_PSS + UNCOUNTED_SHMEM ))

        if [ "$SHMEM_CANDIDATE" -gt "$TOTAL_PSS" ]; then
            TOTAL_PSS=$SHMEM_CANDIDATE
        fi
    fi

    # Percent of total RAM — scaled integer arithmetic, no fork.
    PERCENT_MEM="0.00"
    if [ "$TOTAL_MEM_KB" -gt 0 ] && [ "$TOTAL_PSS" -gt 0 ]; then
        _pct=$(( TOTAL_PSS * 10000 / TOTAL_MEM_KB ))
        printf -v _pct_dec '%02d' $(( _pct % 100 ))
        PERCENT_MEM="$(( _pct / 100 )).$_pct_dec"
    fi

    echo "$TOTAL_PSS, $PERCENT_MEM" 2>/dev/null

    sleep "$INTERVAL"
done
