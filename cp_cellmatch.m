load('~/cellmatch/control_points.mat')
%%
cpstruct = cpselect(moving, fixed, movingPoints, fixedPoints)
%%
mytform= fitgeotrans(movingPoints, fixedPoints, 'polynomial', 3);
%%
warped=imwarp(moving, mytform, 'Outputview', imref2d([789, 789]));
imshowpair(fixed, warped)
%imshow(c)
%%