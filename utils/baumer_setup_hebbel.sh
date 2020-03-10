#!/bin/sh

echo "ATTENTION: Please make sure that your network-manager does not interfere with your network interface configuration"

ip addr add 169.254.0.1/24 broadcast 169.254.0.255 dev eno1
ip addr add 169.254.1.1/24 broadcast 169.254.1.255 dev eno2

echo "Camera IPs configured."

