% -----------------------------
% Dynamic_KB.pl - Knowledge Base for Robot Navigation & Grasping
% -----------------------------

:- discontiguous object/1.
:- discontiguous color/2.
:- discontiguous is_fixed/1.
:- discontiguous is_movable/1.
:- discontiguous can_move/1.
:- discontiguous can_pick/1.
:- discontiguous has_sensor/2.
:- discontiguous has_arm/1.
:- discontiguous position/4.
:- discontiguous robot_link/1.
:- discontiguous step_size/1.
:- discontiguous move_towards/2.
:- discontiguous navigate_to_table/0.
:- discontiguous success/0.
:- dynamic position/4.
:- dynamic robot_position_history/3.

% -----------------------------
% Objects
% -----------------------------

object(robot).
position(robot, 0, 0, 0).
color(robot, blue).
can_move(robot).
has_sensor(robot, camera).
has_sensor(robot, lidar).
has_arm(robot).

object(table).
color(table, brown).
is_fixed(table).
position(table, -2.4597880764764053, 3.810680761172585, 0).

object(obstacle0).
color(obstacle0, [0,0,1,1]).
is_fixed(obstacle0).
position(obstacle0, -4.312774917335269, -0.13927049031850913, 0.2).

object(obstacle1).
color(obstacle1, [1,0.75,0.8,1]).
is_fixed(obstacle1).
position(obstacle1, -2.3382200396161816, -1.8138178180376205, 0.2).

object(obstacle2).
color(obstacle2, [1,0.64,0,1]).
is_fixed(obstacle2).
position(obstacle2, 1.328421119837981, 0.6469845293696164, 0.2).

object(obstacle3).
color(obstacle3, [1,1,0,1]).
is_fixed(obstacle3).
position(obstacle3, -1.0808834760359796, -2.287392696454428, 0.2).

object(obstacle4).
color(obstacle4, [0,0,0,1]).
is_fixed(obstacle4).
position(obstacle4, 0.4569535807778964, -3.686277392493299, 0.2).

object(target).
color(target, red).
is_movable(target).
position(target, -2.4597880764764053, 3.810680761172585, 0.06).
can_pick(target).

% Robot links
robot_link(base_link).
robot_link(fl_wheel).
robot_link(fr_wheel).
robot_link(bl_wheel).
robot_link(br_wheel).
robot_link(rgbd_camera_link).
robot_link(torso_link).
robot_link(lift_carriage).
robot_link(arm_base).
robot_link(shoulder_link).
robot_link(elbow_link).
robot_link(wrist_pitch_link).
robot_link(wrist_roll_link).
robot_link(gripper_base).
robot_link(left_finger).
robot_link(right_finger).

% -----------------------------
% Reasoning Rules
% -----------------------------

% Distance between XY points
distance(X1,Y1,X2,Y2,D) :-
    DX is X2-X1,
    DY is Y2-Y1,
    D is sqrt(DX*DX + DY*DY).

% Fixed objects (obstacles + table)
fixed(X) :-
    object(X),
    is_fixed(X).

% Pickable object
pickable(target).

% Check if any fixed object is in path
obstacle_in_path(Rx,Ry,Tx,Ty) :-
    fixed(O),
    position(O, Ox, Oy, _),
    on_line(Rx,Ry,Tx,Ty,Ox,Oy).

% Safe on-line check with tolerance
on_line(X1,Y1,X2,Y2,Px,Py) :-
    Tolerance = 0.3,
    DX is X2-X1,
    DY is Y2-Y1,
    ( DX =:= 0 -> abs(Px-X1) < Tolerance, Py >= min(Y1,Y2), Py =< max(Y1,Y2)
    ; DY =:= 0 -> abs(Py-Y1) < Tolerance, Px >= min(X1,X2), Px =< max(X1,X2)
    ; T is (Px-X1)/DX,
      T >= 0, T =< 1,
      YLine is Y1 + T*DY,
      abs(YLine-Py) < Tolerance
    ).

% Can robot grasp target
can_grasp(target) :-
    position(robot,Rx,Ry,_),
    position(target,Ox,Oy,_),
    distance(Rx,Ry,Ox,Oy,D),
    D < 0.5.  % grasp if within 0.5m

% -----------------------------
% Movement Logic
% -----------------------------
step_size(0.3).   % 30 cm per step
avoid_step(0.3).  % zig-zag step

% Move robot one step towards target with zig-zag if blocked and record history
move_towards(X,Y) :-
    position(robot,Rx,Ry,Z),
    DX is X-Rx,
    DY is Y-Ry,
    step_size(S),
    (abs(DX) < S -> NewX = X ; NewX is Rx + S*sign(DX)),
    (abs(DY) < S -> NewY = Y ; NewY is Ry + S*sign(DY)),
    ( \+ obstacle_in_path(Rx,Ry,NewX,NewY) ->
        retract(position(robot,Rx,Ry,Z)),
        assert(position(robot,NewX,NewY,Z)),
        assert(robot_position_history(NewX,NewY,Z))
    ; % path blocked → zig-zag
      avoid_step(A),
      (DX =\= 0 -> TempX = Rx, TempY is Ry + A*sign(DY)
      ; TempX is Rx + A*sign(DX), TempY = Ry),
      \+ obstacle_in_path(Rx,Ry,TempX,TempY),
      retract(position(robot,Rx,Ry,Z)),
      assert(position(robot,TempX,TempY,Z)),
      assert(robot_position_history(TempX,TempY,Z))
    ).

% Navigate to table (recursive)
navigate_to_table :-
    position(table,Tx,Ty,_),
    position(robot,Rx,Ry,_),
    distance(Rx,Ry,Tx,Ty,D),
    D > 0.5,
    move_towards(Tx,Ty),
    navigate_to_table.
navigate_to_table.  % stop when close enough

% Success check
success :-
    navigate_to_table,
    can_grasp(target).

% -----------------------------
% Debug Helpers
% -----------------------------
list_objects :-
    object(X), write(X), nl, fail.
list_objects.

list_pickable :-
    pickable(X), write(X), nl, fail.
list_pickable.

list_reachable :-
    pickable(X), write(X), nl, fail.

% Print robot path
print_robot_path :-
    robot_position_history(X,Y,Z),
    write((X,Y,Z)), nl,
    fail.
print_robot_path.
