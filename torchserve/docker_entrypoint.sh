#!/bin/bash

LIBGOMP_PATH=$(find /home/venv/lib/python3.9/site-packages/ -name "libgomp.so.1")

if [ -z "$LIBGOMP_PATH" ]; then
    echo "WARNING: libgomp.so.1 not found. Starting TorchServe without LD_PRELOAD."
else
    echo "Found libgomp.so.1 at: $LIBGOMP_PATH"
    export LD_PRELOAD=$LIBGOMP_PATH
fi

exec "$@"