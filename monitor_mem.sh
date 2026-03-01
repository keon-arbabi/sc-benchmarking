#!/usr/bin/env bash
# Internal memory monitor for MemoryTimer (utils_local.py).
# Polls the target process and its direct children at a fixed interval,
# printing "kb, %" each sample. Killed externally by MemoryTimer after the
# measured operation completes.
#
# Usage: monitor_mem.sh -p PID [-i INTERVAL_SECONDS]

DEFAULT_INTERVAL=0.01

# Returns PIDs of all direct children of a process by reading the kernel's
# task/*/children files. Python multiprocessing workers are always direct
# children (forked/spawned from the main process), so one level is sufficient.
get_child_pids() {
    cat /proc/$1/task/*/children 2>/dev/null | tr ' ' '\n' | grep -v '^$'
}

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

# Baseline system-wide shared memory (tmpfs-backed pages from /dev/shm, shmem, etc.).
# multiprocessing.heap.Arena creates files in /dev/shm and mmaps them; workers fault
# pages in, but main's page table stays empty until first access — so those pages
# don't appear in any process's PSS after workers die. The Shmem: delta captures them.
SHMEM_BASELINE=$(awk '/^Shmem:/ {print $2; exit}' /proc/meminfo)

while true; do
    # Exit cleanly if the monitored process has already ended
    if [ ! -d "/proc/$TARGET_PID" ]; then
        exit 0
    fi

    # Read main process PSS and its Pss_Shmem component in one awk pass.
    # smaps_rollup is ~6x faster than smaps and gives the same totals.
    MAIN_PSS=0; MAIN_PSS_SHMEM=0
    if [ -f "/proc/$TARGET_PID/smaps_rollup" ]; then
        read -r MAIN_PSS MAIN_PSS_SHMEM < <(awk '
            /^Pss:/       { pss   = $2 }
            /^Pss_Shmem:/ { shmem = $2 }
            END           { print pss+0, shmem+0 }
        ' "/proc/$TARGET_PID/smaps_rollup" 2>/dev/null)
    fi

    # Sum PSS across all direct children.
    # PSS distributes shared pages proportionally, so summing across the whole
    # process tree counts shared memory exactly once.
    CHILDREN_PSS=0
    for pid in $(get_child_pids $TARGET_PID); do
        if [ -f "/proc/$pid/smaps_rollup" ]; then
            PSS=$(awk '/^Pss:/ {print $2; exit}' "/proc/$pid/smaps_rollup" 2>/dev/null)
            [ -n "$PSS" ] && CHILDREN_PSS=$((CHILDREN_PSS + PSS))
        fi
    done
    TOTAL_PSS=$((MAIN_PSS + CHILDREN_PSS))

    # Shmem-corrected candidate: covers the case where workers loaded data into
    # tmpfs-backed shared memory but main hasn't touched those pages yet (so they
    # don't appear in main's PSS). UNCOUNTED_SHMEM is the portion of new Shmem not
    # already reflected in main's Pss_Shmem, preventing double-counting once main
    # does access the pages (e.g. during QC/PCA after loading completes).
    SHMEM_CURRENT=$(awk '/^Shmem:/ {print $2; exit}' /proc/meminfo)
    SHMEM_DELTA=$(( SHMEM_CURRENT - SHMEM_BASELINE ))
    [ "$SHMEM_DELTA" -lt 0 ] && SHMEM_DELTA=0
    UNCOUNTED_SHMEM=$(( SHMEM_DELTA - MAIN_PSS_SHMEM ))
    [ "$UNCOUNTED_SHMEM" -lt 0 ] && UNCOUNTED_SHMEM=0
    SHMEM_CANDIDATE=$(( MAIN_PSS + UNCOUNTED_SHMEM ))

    # Report whichever metric is larger:
    # - TOTAL_PSS: correct for private memory, worker memory, and operations after
    #   main has already touched the shared data (QC, PCA, DE, etc.)
    # - SHMEM_CANDIDATE: correct for multi-threaded loading where workers wrote data
    #   but main's page table is still empty
    if [ "$SHMEM_CANDIDATE" -gt "$TOTAL_PSS" ]; then
        TOTAL_PSS=$SHMEM_CANDIDATE
    fi

    PERCENT_MEM="0.0"
    if [ "$TOTAL_MEM_KB" -gt 0 ] && [ "$TOTAL_PSS" -gt 0 ]; then
        PERCENT_MEM=$(awk -v t="$TOTAL_PSS" -v m="$TOTAL_MEM_KB" 'BEGIN {printf "%.2f", (t/m)*100}')
    fi

    echo "$TOTAL_PSS, $PERCENT_MEM" 2>/dev/null

    sleep "$INTERVAL"
done
