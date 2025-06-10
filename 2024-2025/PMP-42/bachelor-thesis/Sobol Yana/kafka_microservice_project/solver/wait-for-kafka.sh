#!/bin/sh

host="$1"
port="$2"
shift 2
cmd="$@"

until nc -z "$host" "$port"; do
  echo "Очікуємо $host:$port..."
  sleep 1
done

echo "$host:$port доступний — запускаємо сервіс!"
exec $cmd
