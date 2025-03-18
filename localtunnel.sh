until lt --port 6006; do
    echo "LocalTunnel crashed with exit code $?. Restarting..." >&2
    sleep 2
done &

