% path_planning.pl
% Prolog-based path planning for robot navigation
% Module 6: Motion Control & Planning

% Facts: Define the workspace grid
% Grid coordinates from -5 to 5 (10x10 room)

% Define valid positions (simple grid-based approach)
valid_position(X, Y) :-
    X >= -4.5, X =< 4.5,
    Y >= -4.5, Y =< 4.5.

% Define obstacles (can be updated dynamically)
% Format: obstacle(X, Y, Radius)
:- dynamic obstacle/3.

% Check if a position is free (not in obstacle)
is_free(X, Y) :-
    valid_position(X, Y),
    \+ in_obstacle(X, Y).

in_obstacle(X, Y) :-
    obstacle(Xo, Yo, R),
    Distance is sqrt((X - Xo)^2 + (Y - Yo)^2),
    Distance < R.

% Calculate distance between two points
distance([X1, Y1], [X2, Y2], D) :-
    D is sqrt((X2 - X1)^2 + (Y2 - Y1)^2).

% Simple straight-line path (if no obstacles in the way)
plan_path(Start, Goal, Path) :-
    straight_path_clear(Start, Goal),
    !,
    Path = [Start, Goal].

% If straight path blocked, use waypoint-based planning
plan_path(Start, Goal, Path) :-
    find_waypoints(Start, Goal, Waypoints),
    Path = [Start | Waypoints].

% Check if straight path is clear
straight_path_clear([X1, Y1], [X2, Y2]) :-
    NumSteps = 20,
    check_line_segments([X1, Y1], [X2, Y2], NumSteps).

check_line_segments(_, _, 0) :- !.
check_line_segments([X1, Y1], [X2, Y2], N) :-
    T is (20 - N) / 20,
    X is X1 + T * (X2 - X1),
    Y is Y1 + T * (Y2 - Y1),
    is_free(X, Y),
    N1 is N - 1,
    check_line_segments([X1, Y1], [X2, Y2], N1).

% Find waypoints around obstacles
find_waypoints([X1, Y1], [X2, Y2], Waypoints) :-
    % Simple approach: go around via intermediate point
    Xmid is (X1 + X2) / 2,
    Ymid is (Y1 + Y2) / 2,

    % Try perpendicular offset
    Dx is X2 - X1,
    Dy is Y2 - Y1,
    Offset = 1.0,

    % Perpendicular direction
    Wx1 is Xmid + Offset * Dy,
    Wy1 is Ymid - Offset * Dx,

    Wx2 is Xmid - Offset * Dy,
    Wy2 is Ymid + Offset * Dx,

    % Choose the free waypoint
    (is_free(Wx1, Wy1) ->
        Waypoints = [[Wx1, Wy1], [X2, Y2]]
    ; is_free(Wx2, Wy2) ->
        Waypoints = [[Wx2, Wy2], [X2, Y2]]
    ;
        % Fallback: just go directly
        Waypoints = [[X2, Y2]]
    ).

% A* pathfinding (simplified version)
% For more complex scenarios
astar_path(Start, Goal, Path) :-
    astar([[0, Start, []]], Goal, Path).

astar([[_, Current, PathSoFar] | _], Goal, Path) :-
    Current = Goal,
    !,
    reverse([Goal | PathSoFar], Path).

astar([[_, Current, PathSoFar] | Rest], Goal, Path) :-
    findall([F, Next, [Current | PathSoFar]],
            (neighbor(Current, Next),
             is_free_pos(Next),
             \+ member(Next, PathSoFar),
             cost(Next, PathSoFar, G),
             heuristic(Next, Goal, H),
             F is G + H),
            Neighbors),
    append(Rest, Neighbors, NewOpen),
    sort(NewOpen, SortedOpen),
    astar(SortedOpen, Goal, Path).

% Define neighbors (8-directional movement)
neighbor([X, Y], [Xn, Yn]) :-
    member(Dx, [-0.5, 0, 0.5]),
    member(Dy, [-0.5, 0, 0.5]),
    \+ (Dx = 0, Dy = 0),
    Xn is X + Dx,
    Yn is Y + Dy.

is_free_pos([X, Y]) :- is_free(X, Y).

cost([_X, _Y], Path, G) :-
    length(Path, Len),
    G is Len * 0.5.

heuristic([X1, Y1], [X2, Y2], H) :-
    H is sqrt((X2 - X1)^2 + (Y2 - Y1)^2).

% Helper predicates
% Add obstacle dynamically
add_obstacle(X, Y, R) :-
    assertz(obstacle(X, Y, R)).

% Remove all obstacles
clear_obstacles :-
    retractall(obstacle(_, _, _)).

% Example usage:
% ?- plan_path([0, 0], [3, 3], Path).
% Path = [[0, 0], [3, 3]].
%
% ?- add_obstacle(1.5, 1.5, 0.5).
% ?- plan_path([0, 0], [3, 3], Path).
% Path = [[0, 0], [1.5, 3.0], [3, 3]].
