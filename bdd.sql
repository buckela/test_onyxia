-- ============================================================
-- Schéma de référence — bataille navale (PostgreSQL)
-- NB : il est créé automatiquement au démarrage par bdd.init_schema().
--      Ce fichier sert de documentation / provisioning manuel.
-- ============================================================

CREATE TABLE IF NOT EXISTS players (
    player_id   SERIAL PRIMARY KEY,
    pseudo      TEXT UNIQUE NOT NULL,
    online      BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS games (
    game_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player1     TEXT NOT NULL REFERENCES players(pseudo),
    player2     TEXT REFERENCES players(pseudo),   -- NULL = file d'attente
    status      TEXT NOT NULL DEFAULT 'waiting'
                CHECK (status IN ('waiting', 'playing', 'finished')),
    winner      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);
-- File de matchmaking : recherche rapide des parties en attente
CREATE INDEX IF NOT EXISTS idx_games_waiting
    ON games (created_at) WHERE status = 'waiting';

-- ships : [{"x": 0, "y": 0, "ship": "Torpilleur"}, ...]
CREATE TABLE IF NOT EXISTS boards (
    game_id     UUID NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player      TEXT NOT NULL,
    ships       JSONB NOT NULL,
    PRIMARY KEY (game_id, player)
);

CREATE TABLE IF NOT EXISTS moves (
    move_id     BIGSERIAL PRIMARY KEY,
    game_id     UUID NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    player      TEXT NOT NULL,
    turn        INT NOT NULL,
    x           INT NOT NULL CHECK (x BETWEEN 0 AND 9),
    y           INT NOT NULL CHECK (y BETWEEN 0 AND 9),
    result      TEXT NOT NULL CHECK (result IN ('miss', 'hit', 'sunk')),
    ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_moves_game ON moves (game_id, move_id);