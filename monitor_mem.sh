#!/bin/bash

# Memory monitor for single-cell analysis - tracks process trees
DEFAULT_INTERVAL=0.1

cleanup() { exit 0; }

# Get RSS from /proc/PID/status
get_rss() {
    [ -r "/proc/$1/status" ] && \
        awk '/^VmRSS:/ {print $2+0; exit}' "/proc/$1/status" 2>/dev/null || \
        echo "0"
}

# Get all PIDs in process tree using ps
get_process_tree() {
    local root_pid=$1 all_pids="$1" to_check="$1" depth=0
    
    while [ -n "$to_check" ] && [ $depth -lt 10 ]; do
        local new_pids=""
        for pid in $to_check; do
            local children=$(ps --no-headers -o pid --ppid $pid 2>/dev/null | \
                           tr '\n' ' ')
            [ -n "$children" ] && new_pids="$new_pids $children"
        done
        
        [ -n "$new_pids" ] && {
            all_pids="$all_pids $new_pids"
            to_check="$new_pids"
        } || break
        ((depth++))
    done
    
    echo "$all_pids" | tr ' ' '\n' | sort -u | tr '\n' ' '
}

# Sum memory across process tree
get_total_memory() {
    local total_mem=0
    for pid in $(get_process_tree $1); do
        if [ -d "/proc/$pid" ]; then
            local mem=$(get_rss $pid)
            total_mem=$((total_mem + mem))
        fi
    done
    echo $total_mem
}

# Parse arguments
TARGET_PID="" INTERVAL="$DEFAULT_INTERVAL"
[[ "$1" =~ ^[0-9]+$ ]] && { TARGET_PID="$1"; shift; }

while getopts ":p:i:" opt; do
    case $opt in
        p) TARGET_PID="$OPTARG" ;;
        i) INTERVAL="$OPTARG" ;;
        *) echo "Usage: $0 [-p PID] [-i INTERVAL]" >&2; exit 1 ;;
    esac
done

[ -z "$TARGET_PID" ] && { 
    echo "Error: PID required" >&2; exit 1; 
}
[ ! -d "/proc/$TARGET_PID" ] && { 
    echo "Error: PID $TARGET_PID not found" >&2; exit 1; 
}

trap cleanup SIGINT SIGTERM
TOTAL_MEM_KB=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)

# Main loop
while [ -d "/proc/$TARGET_PID" ]; do
    TOTAL_KB=$(get_total_memory $TARGET_PID)
    PERCENT="0.00"
    [ "$TOTAL_MEM_KB" -gt 0 ] && [ "$TOTAL_KB" -gt 0 ] && \
        PERCENT=$(awk -v m="$TOTAL_KB" -v t="$TOTAL_MEM_KB" \
                  'BEGIN {printf "%.2f", (m/t)*100}')
    echo "$TOTAL_KB, $PERCENT"
    sleep "$INTERVAL"
done 