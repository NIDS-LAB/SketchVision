#!/bin/bash

ip link add name veth_a type veth peer name veth_b

ip link set dev veth_a up
ip link set dev veth_b up

ip addr add 10.10.10.1/24 dev veth_a
ip addr add 10.10.10.2/24 dev veth_b

#ip link del veth_a

#ip link set dev veth_a xdpgeneric off
