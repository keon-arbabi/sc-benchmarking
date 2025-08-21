#!/bin/bash

# Configuration
DEFAULT_INTERVAL=0.25
MONITOR_PID_FILE="/tmp/memory_monitor_script.pid"

cleanup() {
    rm -f "$MONITOR_PID_FILE"
    exit 0
}

# Recursively find all descendant PIDs
get_process_tree() {
    local parent=$1
    local pids="$parent"
    local current="$parent"
    
    while [ -n "$current" ]; do
        local new=""
        for pid in $current; do
            if [ -d "/proc/$pid/task" ]; then
                for task in /proc/$pid/task/*/children; do
                    [ -r "$task" ] && new="$new $(cat "$task" 2>/dev/null)"
                done
            fi
        done
        current="$new"
        [ -n "$new" ] && pids="$pids $new"
    done
    echo "$pids" | tr ' ' '\n' | sort -u | tr '\n' ' '
}

# Get PSS (Proportional Set Size) from /proc/PID/smaps
get_pss() {
    [ -r "/proc/$1/smaps" ] && \
        awk '/^Pss:/ {sum += $2} END {print sum+0}' "/proc/$1/smaps" 2>/dev/null || echo "0"
}

# Get RSS from /proc/PID/status as fallback
get_rss() {
    [ -r "/proc/$1/status" ] && awk '
        /^Rss(Anon|File|Shmem):/ {sum += $2} 
        END {print sum+0}
    ' "/proc/$1/status" 2>/dev/null || echo "0"
}

# Get shared memory usage in KB
get_shm_kb() {
    df /dev/shm 2>/dev/null | tail -1 | awk '{print $3+0}'
}

# Parse arguments
TARGET_PID=""
INTERVAL="$DEFAULT_INTERVAL"

# Handle positional PID for backward compatibility
[[ "$1" =~ ^[0-9]+$ ]] && { TARGET_PID="$1"; shift; }

while getopts ":p:i:" opt; do
    case $opt in
        p) TARGET_PID="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        *) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

[ -z "$TARGET_PID" ] && { echo "Error: PID required." >&2; exit 1; }
[ ! -d "/proc/$TARGET_PID" ] && \
    { echo "Error: PID $TARGET_PID not found." >&2; exit 1; }

# Setup
echo $$ > "$MONITOR_PID_FILE"
trap cleanup SIGINT SIGTERM

# Store initial shared memory
INITIAL_SHM=$(get_shm_kb)

# Main monitoring loop
while [ -d "/proc/$TARGET_PID" ]; do
    # Sum memory for all processes in tree
    total_mem=0
    for pid in $(get_process_tree "$TARGET_PID"); do
        if [ -d "/proc/$pid" ]; then
            pss=$(get_pss "$pid")
            if [ "$pss" -gt 0 ]; then
                total_mem=$((total_mem + pss))
            else
                rss=$(get_rss "$pid")
                total_mem=$((total_mem + rss))
            fi
        fi
    done
    
    # Add shared memory delta
    shm_delta=$(($(get_shm_kb) - INITIAL_SHM))
    [ "$shm_delta" -gt 0 ] && total_mem=$((total_mem + shm_delta))
    
    # Calculate percentage
    total_system=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    percent="0.0"
    [ "$total_system" -gt 0 ] && [ "$total_mem" -gt 0 ] && \
        percent=$(awk -v m="$total_mem" -v t="$total_system" \
                  'BEGIN {printf "%.2f", (m/t)*100}')
    
    echo "$total_mem, $percent"
    sleep "$INTERVAL"
done

cleanup