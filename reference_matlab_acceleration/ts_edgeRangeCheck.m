function eId = ts_edgeRangeCheck(edge, vrtx, ceId, gmp, rg)
% Vectorized candidate search for the author MATLAB/Octave T4 transition.
% The periodic shifts and ascending candidate order match the original.

rows = edge{1};
column = rg.ei(5);
query = rows(ceId, column:column+3);
endpoints = rows(ceId, rg.ei(1):rg.ef(1));
incident = vrtx(endpoints, rg.vi(1):rg.vf(1));
eligible = rows(:,3) == 0 & rows(:,4) ~= rows(ceId,4);
valid = incident >= 1 & incident <= numel(eligible) & incident == fix(incident);
eligible(incident(valid)) = false;
candidate = find(eligible);

box = gmp.bs;
shearOffset = box * gmp.sstn;
[minX, minY] = unwrapNext(rows(candidate,column), rows(candidate,column+2), ...
                         query(1), query(3), box, shearOffset);
[maxX, maxY] = unwrapNext(rows(candidate,column+1), rows(candidate,column+3), ...
                         minX, minY, box, shearOffset);
overlap = maxX >= query(1) & minX <= query(2) ...
        & maxY >= query(3) & minY <= query(4);
eId = candidate(overlap);
end

function [x, y] = unwrapNext(x, y, previousX, previousY, box, shearOffset)
% Match ts_crdLocal's four ordered tests and single-box corrections.
half = box / 2;
move = y - previousY > half;
y(move) = y(move) - box;
x(move) = x(move) - shearOffset;
move = y - previousY < -half;
y(move) = y(move) + box;
x(move) = x(move) + shearOffset;
move = x - previousX > half;
x(move) = x(move) - box;
move = x - previousX < -half;
x(move) = x(move) + box;
end
