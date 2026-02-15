% Path Planning Module for Robot Navigation
% Prolog rules for simple path planning

% Define grid connections (adjacency)
connected(X1, Y1, X2, Y2) :-
    abs(X1 - X2) =< 1,
    abs(Y1 - Y2) =< 1,
    \+ (X1 = X2, Y1 = Y2).

% Path finding using A* (simplified)
plan_path(Start, Goal, Path) :-
    astar([Start], Goal, [], Path).

% A* search implementation
astar([Current|_], Goal, Visited, [Current|[]]) :-
    Current = Goal.

astar([Current|Rest], Goal, Visited, [Current|Path]) :-
    Current \= Goal,
    findall(Neighbor, (connected_state(Current, Neighbor), \+ member(Neighbor, Visited)), Neighbors),
    append(Rest, Neighbors, NewOpen),
    astar(NewOpen, Goal, [Current|Visited], Path).

% Helper: connected states
connected_state([X1, Y1], [X2, Y2]) :-
    connected(X1, Y1, X2, Y2).

% Obstacle checking
is_free([X, Y]) :-
    \+ obstacle([X, Y]).

% Dynamic obstacle facts (will be asserted at runtime)
:- dynamic obstacle/1.
:- dynamic start_pos/1.
:- dynamic goal_pos/1.
