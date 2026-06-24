#!/usr/bin/env bash
# Creates one dedicated database + user per platform component.
# Runs automatically as part of the postgres container initialization.
set -euo pipefail

create_db_and_user() {
    local db=$1
    local user=$2
    local pass=$3
    echo "[init-db] Creating database='$db' user='$user'"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$user') THEN
                CREATE ROLE $user WITH LOGIN PASSWORD '$pass';
            END IF;
        END
        \$\$;
        CREATE DATABASE $db OWNER $user;
        GRANT ALL PRIVILEGES ON DATABASE $db TO $user;
SQL
}

create_db_and_user airflow   airflow   airflow123
create_db_and_user mlflow    mlflow    mlflow123
create_db_and_user superset  superset  superset123
create_db_and_user keycloak  keycloak  keycloak123

echo "[init-db] All databases initialized."
