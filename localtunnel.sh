until lt --port 6006 | tee lt.log; do
    echo "LocalTunnel crashed with exit code $?. Restarting..." >&2
    sleep 2
done &
