#!/usr/bin/env bash

DEFAULT_INTERVAL=0.01
MONITOR_PID_FILE="/tmp/memory_monitor_script.pid"

# Function to clean up the PID file on exit
cleanup() {
    rm -f "$MONITOR_PID_FILE"
    exit 0
}

# Function to get immediate child PIDs of a given parent PID
get_child_pids() {
    local ppid=$1
    pgrep -P "$ppid"
}

# Function to recursively get all descendant PIDs
get_all_descendant_pids() {
    local ppid=$1
    local children=$(get_child_pids "$ppid")
    for pid in $children; do
        echo "$pid"
        get_all_descendant_pids "$pid"
    done
}

TARGET_PID=""
INTERVAL="$DEFAULT_INTERVAL"

# Handle positional PID for backward compatibility
if [[ "$1" =~ ^[0-9]+$ ]]; then
    TARGET_PID="$1"
    shift
fi

while getopts ":p:i:h" opt; do
    case ${opt} in
        p ) TARGET_PID=$OPTARG ;;
        i ) INTERVAL=$OPTARG ;;
        h ) usage ;;
        \? ) echo "Invalid option: $OPTARG" 1>&2; usage ;;
        : ) echo "Invalid option: $OPTARG requires an argument" 1>&2; usage ;;
    esac
done

if [ -z "$TARGET_PID" ]; then
    echo "Error: You must specify a PID."
    usage
fi

# Check if target PID exists initially
if [ ! -d "/proc/$TARGET_PID" ]; then
    echo "Error: Process with PID $TARGET_PID not found."
    exit 1
fi

# Store the PID of this monitoring script and set up cleanup
echo $$ > "$MONITOR_PID_FILE"
trap cleanup SIGINT SIGTERM

while true; do
    # Check if the target process still exists
    if [ ! -d "/proc/$TARGET_PID" ]; then
        cleanup
    fi

    ALL_PIDS="$TARGET_PID $(get_all_descendant_pids $TARGET_PID)"
    TOTAL_PSS_SUM=0

    for pid in $ALL_PIDS; do
        if [ -f "/proc/$pid/smaps" ]; then
            # Sum up the PSS values from the smaps file
            PSS=$(awk '/^Pss:/ {sum+=$2} END {print sum}' "/proc/$pid/smaps" 2>/dev/null)
            if [ -n "$PSS" ]; then
                TOTAL_PSS_SUM=$((TOTAL_PSS_SUM + PSS))
            fi
        fi
    done

    # Calculate memory percentage
    TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    PERCENT_MEM="0.0"
    if [ "$TOTAL_MEM_KB" -gt 0 ] && [ "$TOTAL_PSS_SUM" -gt 0 ]; then
        PERCENT_MEM=$(awk -v pss="$TOTAL_PSS_SUM" -v total="$TOTAL_MEM_KB" 'BEGIN {printf "%.2f", (pss/total)*100}')
    fi

    # Output: PSS_in_KiB, Percentage
    echo "$TOTAL_PSS_SUM, $PERCENT_MEM"

    sleep "$INTERVAL"
done
