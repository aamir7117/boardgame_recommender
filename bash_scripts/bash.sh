#!/bin/bash
sudo killall firefox
sleep 3

sudo Xvfb :10 -ac &

sleep 3

sudo kill -9 888

sudo export DISPLAY=:10

sleep 2

firefox &

sleep 3
sudo kill -9 888
