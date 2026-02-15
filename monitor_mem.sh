#!/usr/bin/env bash

DEFAULT_INTERVAL=0.01
MONITOR_PID_FILE="/tmp/memory_monitor_script.pid"

cleanup() {
    rm -f "$MONITOR_PID_FILE"
    exit 0
}

# Get direct children only
get_child_pids() {
    cat /proc/$1/task/*/children 2>/dev/null | tr ' ' '\n' | grep -v '^$'
}

TARGET_PID=""
INTERVAL="$DEFAULT_INTERVAL"

if [[ "$1" =~ ^[0-9]+$ ]]; then
    TARGET_PID="$1"
    shift
fi

while getopts ":p:i:h" opt; do
    case ${opt} in
        p ) TARGET_PID=$OPTARG ;;
        i ) INTERVAL=$OPTARG ;;
        h ) echo "Usage: $0 <PID> [-i interval]"; exit 0 ;;
        \? ) echo "Invalid option: $OPTARG" 1>&2; exit 1 ;;
        : ) echo "Option $OPTARG requires an argument" 1>&2; exit 1 ;;
    esac
done

if [ -z "$TARGET_PID" ]; then
    echo "Error: You must specify a PID."
    exit 1
fi

if [ ! -d "/proc/$TARGET_PID" ]; then
    echo "Error: Process with PID $TARGET_PID not found."
    exit 1
fi

echo $$ > "$MONITOR_PID_FILE"
trap cleanup SIGINT SIGTERM

# Baseline /dev/shm usage (for shared memory tracking)
BASELINE_SHM=$(df /dev/shm 2>/dev/null | awk 'NR==2 {print $3}')
TOTAL_MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)

while true; do
    if [ ! -d "/proc/$TARGET_PID" ]; then
        cleanup
    fi

    ALL_PIDS="$TARGET_PID $(get_child_pids $TARGET_PID)"
    TOTAL_PSS=0

    for pid in $ALL_PIDS; do
        # Use smaps_rollup (6x faster than smaps)
        if [ -f "/proc/$pid/smaps_rollup" ]; then
            PSS=$(awk '/^Pss:/ {print $2; exit}' "/proc/$pid/smaps_rollup" 2>/dev/null)
            [ -n "$PSS" ] && TOTAL_PSS=$((TOTAL_PSS + PSS))
        fi
    done

    # Track /dev/shm delta (captures multiprocessing shared memory)
    CURRENT_SHM=$(df /dev/shm 2>/dev/null | awk 'NR==2 {print $3}')
    SHM_DELTA=$((CURRENT_SHM - BASELINE_SHM))
    [ $SHM_DELTA -lt 0 ] && SHM_DELTA=0

    # Total memory = PSS + shared memory delta
    TOTAL=$((TOTAL_PSS + SHM_DELTA))

    PERCENT_MEM="0.0"
    if [ "$TOTAL_MEM_KB" -gt 0 ] && [ "$TOTAL" -gt 0 ]; then
        PERCENT_MEM=$(awk -v t="$TOTAL" -v m="$TOTAL_MEM_KB" 'BEGIN {printf "%.2f", (t/m)*100}')
    fi

    echo "$TOTAL, $PERCENT_MEM" 2>/dev/null

    sleep "$INTERVAL"
done
