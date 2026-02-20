% -----------------------------
% Dynamic_KB.pl - Knowledge Base for Robot Navigation & Grasping
% -----------------------------

:- discontiguous object/1.
:- discontiguous color/2.
:- discontiguous is_fixed/1.
:- discontiguous is_movable/1.
:- discontiguous can_pick/1.
:- discontiguous position/4.
:- discontiguous sensor/2.
:- discontiguous sensor_link_for_robot/2.
:- dynamic robot_position_history/3.
:- dynamic position/4.
:- dynamic obstacle_fixed/1.  % Tracks if an obstacle has been "placed"

% -----------------------------
% Objects
% -----------------------------
object(table).
object(target).
object(obstacle0).
object(obstacle1).
object(obstacle2).
object(obstacle3).
object(obstacle4).
object(robot).

color(robot, blue).
color(table, brown).
color(target, red).
color(obstacle0, [0,0,1,1]).
color(obstacle1, [1,0.75,0.8,1]).
color(obstacle2, [1,0.64,0,1]).
color(obstacle3, [1,1,0,1]).
color(obstacle4, [0,0,0,1]).

is_fixed(table).  
is_fixed(target). 
is_fixed(obstacle0). 
is_fixed(obstacle1).
is_fixed(obstacle2).
is_fixed(obstacle3).
is_fixed(obstacle4).

% -----------------------------
% Utility predicates
% -----------------------------
fixed(X) :- object(X), is_fixed(X).

is_movable(target).
can_pick(target).
can_move(robot).
has_arm(robot).

% -----------------------------
% Robot Links
% -----------------------------
robot_link(base_link).
robot_link(rgbd_camera_link).
robot_link(lidar_link).
robot_link(imu_link).
robot_link(torso_link).
robot_link(arm_base).
robot_link(shoulder_link).
robot_link(elbow_link).
robot_link(wrist_pitch_link).
robot_link(wrist_roll_link).
robot_link(gripper_base).
robot_link(left_finger).
robot_link(right_finger).

% -----------------------------
% Sensors
% -----------------------------
sensor(rgbd_camera_link, rgbd_camera).
sensor(lidar_link, lidar).
sensor(imu_link, imu).
sensor(joint_state_sensor, joint_states).

has_sensor(robot, rgbd_camera_rgb).
has_sensor(robot, rgbd_camera_depth).
has_sensor(robot, rgbd_camera_mask).
has_sensor(robot, lidar).
has_sensor(robot, imu).
has_sensor(robot, joint_states).

sensor_link_for_robot(rgbd_camera_rgb, rgbd_camera_link).
sensor_link_for_robot(rgbd_camera_depth, rgbd_camera_link).
sensor_link_for_robot(rgbd_camera_mask, rgbd_camera_link).
sensor_link_for_robot(lidar, lidar_link).
sensor_link_for_robot(imu, base_link).
sensor_link_for_robot(joint_states, base_link).

% -----------------------------
% Object positions
% -----------------------------
position(robot, 0, 0, 0).
position(target, -2.4598, 3.8106, 0.06). % Fixed on table
position(table, -2.4598, 3.8106, 0).

% Obstacles positions can change every run
position(obstacle0, -4.3127, -0.139, 0.2).
position(obstacle1, -2.3382, -1.8138, 0.2).
position(obstacle2, 1.3284, 0.6469, 0.2).
position(obstacle3, -1.0808, -2.2873, 0.2).
position(obstacle4, 0.4569, -3.6862, 0.2).

position(rgbd_camera_link, 0, 0, 1.2).
position(lidar_link, 0, 0, 0.5).
position(base_link, 0, 0, 0).

% -----------------------------
% Utility predicates
% -----------------------------
distance(X1,Y1,Z1,X2,Y2,Z2,D) :-
    DX is X2-X1,
    DY is Y2-Y1,
    DZ is Z2-Z1,
    D is sqrt(DX*DX + DY*DY + DZ*DZ).

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

% -----------------------------
% Pickable objects
% -----------------------------
pickable(X) :- can_pick(X).

% -----------------------------
% Sensor logic
% -----------------------------
perception_sensor_link(Link) :-
    has_sensor(robot, Stream),
    sensor_link_for_robot(Stream, Link),
    (sensor(Link, rgbd_camera) ; sensor(Link, lidar)).

visible_to_sensor(Link, Obj) :-
    perception_sensor_link(Link),
    object(Obj),
    Obj \= robot,
    position(robot,Rx,Ry,Rz),
    position(Obj,Ox,Oy,Oz),
    distance(Rx,Ry,Rz,Ox,Oy,Oz,D),
    D =< 5.0.   % detection range in meters

detectable_objects(Obj) :-
    has_sensor(robot, Stream),
    sensor_link_for_robot(Stream, Link),
    visible_to_sensor(Link, Obj).

unique_detectable_objects(Objs) :- 
    setof(Obj, detectable_objects(Obj), Objs).

% -----------------------------
% Robot movement & grasping
% -----------------------------
step_size(0.3).
avoid_step(0.3).

move_towards(X,Y) :-
    position(robot,Rx,Ry,Z),
    DX is X-Rx,
    DY is Y-Ry,
    step_size(S),
    (abs(DX) < S -> NewX = X ; NewX is Rx + S*sign(DX)),
    (abs(DY) < S -> NewY = Y ; NewY is Ry + S*sign(DY)),
    (\+ obstacle_in_path(Rx,Ry,NewX,NewY) ->
        retract(position(robot,Rx,Ry,Z)),
        assert(position(robot,NewX,NewY,Z)),
        assert(robot_position_history(NewX,NewY,Z))
    ;
      avoid_step(A),
      (DX =\= 0 -> TempX = Rx, TempY is Ry + A*sign(DY)
      ; TempX is Rx + A*sign(DX), TempY = Ry),
      \+ obstacle_in_path(Rx,Ry,TempX,TempY),
      retract(position(robot,Rx,Ry,Z)),
      assert(position(robot,TempX,TempY,Z)),
      assert(robot_position_history(TempX,TempY,Z))
    ).

obstacle_in_path(Rx,Ry,Tx,Ty) :-
    fixed(O),
    position(O,Ox,Oy,_),
    on_line(Rx,Ry,Tx,Ty,Ox,Oy).

can_grasp(target) :-
    position(robot,Rx,Ry,_),
    position(target,Ox,Oy,_),
    distance(Rx,Ry,0,Ox,Oy,0,D),
    D < 0.5.

navigate_to_table :-
    position(table,Tx,Ty,_),
    position(robot,Rx,Ry,_),
    distance(Rx,Ry,0,Tx,Ty,0,D),
    D > 0.5,
    move_towards(Tx,Ty),
    navigate_to_table.
navigate_to_table.

success :- navigate_to_table, can_grasp(target).

% -----------------------------
% Dynamic updates
% -----------------------------
update_position(Obj,X,Y,Z) :-
    (object(Obj), \+ fixed(Obj) ; (fixed(Obj), \+ obstacle_fixed(Obj))),
    retractall(position(Obj,_,_,_)),
    assert(position(Obj,X,Y,Z)),
    (fixed(Obj) -> assert(obstacle_fixed(Obj)) ; true).

% -----------------------------
% Debug helpers
% -----------------------------
list_objects :- object(X), write(X), nl, fail.
list_objects.

list_pickable :- pickable(X), write(X), nl, fail.
list_pickable.

print_robot_path :- robot_position_history(X,Y,Z), write((X,Y,Z)), nl, fail.
print_robot_path.
