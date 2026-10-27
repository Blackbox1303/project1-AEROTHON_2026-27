#!/bin/bash
# ON THE LENOVO: start / drive the Jetson's ground station from here - for when the dashboard (Jetson :8080) is not up
# after the Jetson's boot. Installed as ~/.local/bin/aerothon by install_lenovo_agent.sh (ssh key to the Jetson).
#   aerothon up                  start the Jetson dashboard if it is down, then open it in the browser
#   aerothon restart             restart the Jetson dashboard (systemd --user gcs-dashboard)
#   aerothon launch [world] [segment]  dashboard up + launch the sim stack; the segment (full | qr | tunnel | qr_drop,
#                                default full) starts by itself once the stack is READY
#   aerothon radio [dev]         THE FINAL FLIGHT: the 8080 dashboard on THIS PC on the SiK radio (default
#                                /dev/ttyUSB0:57600), QGC gets every packet too: http://127.0.0.1:8080
#   aerothon launch-real [link]  dashboard up + launch the real drone's stack (link default /dev/ttyACM0:921600)
#   aerothon stop                stop the stack (and close the Gazebo GUI here)
#   aerothon status              dashboard + stack state
set -o pipefail
JETSON=${JETSON:-szyfrowac@10.42.0.247}
URL=http://${JETSON#*@}:8080
AGENT=http://127.0.0.1:8765
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 "$JETSON")

dash_up() { curl -s -m 3 -o /dev/null "$URL/" ; }

# systemctl --user over a non-login ssh needs the user's bus; the service runs the dashboard (lingering user)
jetson_service() {
    "${SSH[@]}" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); systemctl --user $1 gcs-dashboard" \
        || { echo "ssh to $JETSON failed (Jetson on? key installed? install_lenovo_agent.sh)"; return 1; }
}

ensure_dashboard() {
    dash_up && return 0
    echo "dashboard $URL down: starting it on the Jetson ..."
    jetson_service restart || return 1
    for _ in $(seq 40); do dash_up && { echo "dashboard up: $URL"; return 0; }; sleep 1; done
    echo "the dashboard did not come up in 40 s: on the Jetson see ~/workspaces/isaac_ros-dev/runs/gcs/server.log"
    return 1
}

api() {   # api <what> [json]
    local body='{}'
    [ -n "$2" ] && body=$2
    curl -s -m 200 -X POST -H 'Content-Type: application/json' -d "$body" "$URL/api/$1"; echo
}

case "${1:-up}" in
    up)          ensure_dashboard && { xdg-open "$URL" >/dev/null 2>&1 & } ;;
    restart)     jetson_service restart && sleep 3 && ensure_dashboard ;;
    launch)      ensure_dashboard && api stack_start "{\"mode\": \"sim\", \"world\": \"${2:-competition}\", \"segment\": \"${3:-full}\"}" ;;
    radio)
        # the radio belongs to one program: the old Radio GCS (8081) lets go of it first
        systemctl --user stop radio-gcs 2>/dev/null
        GCS=~/.local/share/aerothon_gcs
        [ -f $GCS/gcs_server.py ] || { echo "$GCS missing: re-run install_lenovo_agent.sh"; exit 1; }
        ( sleep 3; xdg-open http://127.0.0.1:8080 >/dev/null 2>&1 ) &
        echo "radio dashboard: http://127.0.0.1:8080 (Ctrl-C stops it); QGC: UDP 14550 auto-connect, SiK auto-connect OFF"
        exec python3 $GCS/gcs_server.py --radio "${2:-/dev/ttyUSB0:57600}" --port 8080 ;;
    launch-real) ensure_dashboard && api stack_start "{\"mode\": \"real\", \"link\": \"${2:-/dev/ttyACM0:921600}\"}" ;;
    stop)        curl -s -m 3 -X POST "$AGENT/gui/stop" >/dev/null; dash_up && api stack_stop ;;
    status)
        if dash_up; then
            echo "dashboard: up ($URL)"
            curl -s -m 3 "$URL/api/status" -X POST; echo
        else
            echo "dashboard: DOWN ($URL) - 'aerothon up' starts it"
            "${SSH[@]}" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); systemctl --user status gcs-dashboard --no-pager | head -5"
        fi ;;
    *) sed -n 2,11p "$0"; exit 1 ;;
esac
