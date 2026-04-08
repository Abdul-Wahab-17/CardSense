#!/usr/bin/env sh
set -eu

DB_PATH="${1:-data/cardsense.db}"
BACKUP_DIR="${2:-data/backups}"
TS="$(date +%Y%m%d-%H%M%S)"

if [ ! -f "$DB_PATH" ]; then
  echo "db file not found: $DB_PATH"
  exit 1
fi

mkdir -p "$BACKUP_DIR"
BACKUP_FILE="$BACKUP_DIR/cardsense-${TS}.db"
RESTORE_FILE="$BACKUP_DIR/cardsense-restore-test-${TS}.db"

cp "$DB_PATH" "$BACKUP_FILE"
cp "$BACKUP_FILE" "$RESTORE_FILE"

if [ ! -s "$BACKUP_FILE" ] || [ ! -s "$RESTORE_FILE" ]; then
  echo "backup/restore drill failed"
  exit 1
fi

echo "backup created: $BACKUP_FILE"
echo "restore test copy created: $RESTORE_FILE"
echo "backup/restore drill completed"
