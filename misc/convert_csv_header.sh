#!/bin/sh
sed -i 's/bias/Bias Voltage [V]/g' $@
sed -i 's/amps/Alpha Pulse Amplitude [V]/g' $@
sed -i 's/\[V\] error/error \[V\]/g' $@
