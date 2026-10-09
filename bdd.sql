CREATE TABLE games(
    game_id UUID PRIMARY KEY,
    player1_id TEXT,
    player2_id TEXT,
    status TEXT,
    winner_id,
    created_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

CREATE TABLE boards(
    game_id UUID REFERENCES games,
    player_id TEXT,
    ships JSONB, --{"b1":[[0,0],[0,1],[0,2]]}
    PRIMARY KEY (game_id, player1_id)
);

CREATE TABLE move(
    move_id BIGSERIAL PRIMARY KEY,
    game_id UUID,
    player_id TEXT,
    turn INT,
    x int,
    y int,
    result TEXT, --miss, hit, sunk
    ts TIMESTAMPTZ DEFAULT now()
);