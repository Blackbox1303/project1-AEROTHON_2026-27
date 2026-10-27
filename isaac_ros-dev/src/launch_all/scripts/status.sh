#!/bin/bash
# status.sh: what of the sim mission is running (patterns live here, not on the caller's command line)
ps -eo pid,args | grep -E 'sim_mission\.launch|host_all\.launch|save_run\.sh|gz sim|bin/px4|sitl/bin/arducopter --model JSON|container_all\.launch|mavros_node|parameter_bridge' \
    | grep -v grep | cut -c1-110
echo "status: $(ps -eo args | grep -cE 'gz sim|bin/px4|sitl/bin/arducopter --model JSON|container_all\.launch|mavros_node' | awk '{print $1 - 1}') sim/mission processes"
