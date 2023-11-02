#!/bin/bash
while true
do
 python3 DILLEMA_counterfactual.py || echo "Error... restarting..." >&2
 echo "Press Ctrl-C to quit." && sleep 1
done
