%% Adjust vertex position in force direction.

function [vrtxN,edgeN,faceN]=ts_iteration(vrtx,edge,face,rg,gmp,mcp)

[vrtxN,edgeN,faceN]=deal(vrtx,edge,face);
[vMx,eMx]=deal(size(vrtx,1),size(edge{1},1));
vrtxN(:,rg.vi(4):rg.vf(4))=zeros(vMx,2);

% Evaluate all film segments in one batch, keeping each film's endpoint order.
segmentCount=edge{1}(:,rg.ei(4))+1;
lastPoint=cumsum(segmentCount+1);
firstPoint=[1; lastPoint(1:end-1)+1];
points=vertcat(edge{2}{:});
isStart=true(size(points,1),1);
isStart(lastPoint)=false;
start=find(isStart);
finish=start+1;
film=repelem((1:eMx).',segmentCount);
difference=points(finish,:)-points(start,:);
lengths=vecnorm(difference,2,2);
tangent=difference./lengths;
normal=[-tangent(:,2),tangent(:,1)];

% Use the same sequential Lees-Edwards corrections as ts_edgeVector.
center=face{1}(edge{1}(film,4),rg.fi(6):rg.ff(6));
midpoint=(points(finish,:)+points(start,:))/2;
box=gmp.bs;
half=box/2;
shearOffset=box*gmp.sstn;
midpoint=wrapRelative(midpoint,center,box,half,shearOffset);
reverse=sum(normal.*(midpoint-center),2)<0;
normal(reverse,:)=-normal(reverse,:);

areas=face{1}(:,rg.fi(2));
pressure=mcp.del*(1./(areas/gmp.lsc^2)-1);
sides=edge{1}(film,rg.ei(2):rg.ef(2));
normalForce=zeros(size(lengths));
onCell=sides(:,1)~=0;
normalForce(onCell)=normalForce(onCell) ...
    - pressure(sides(onCell,1)).*lengths(onCell)/2/gmp.lsc;
onCell=sides(:,2)~=0;
normalForce(onCell)=normalForce(onCell) ...
    + pressure(sides(onCell,2)).*lengths(onCell)/2/gmp.lsc;
normal=normalForce.*normal;
tangent=max(0,edge{1}(film,rg.ei(3))).*tangent;

% At interior vertices, apply the preceding segment's two forces first.
force=zeros(size(points));
force(finish,:)=force(finish,:)-tangent;
force(finish,:)=force(finish,:)+normal;
force(start,:)=force(start,:)+tangent;
force(start,:)=force(start,:)+normal;
updated=points+mcp.psi*gmp.dt*force;
columns=rg.vi(4):rg.vf(4);
for ii=1:eMx
    idx=firstPoint(ii):lastPoint(ii);
    vrtxN(edgeN{1}(ii,1),columns)=vrtxN(edgeN{1}(ii,1),columns)+force(idx(1),:);
    vrtxN(edgeN{1}(ii,2),columns)=vrtxN(edgeN{1}(ii,2),columns)+force(idx(end),:);
    edgeN{3}{ii}=force(idx,:);
end

vrtxN(:,rg.vi(3):rg.vf(3))=vrtxN(:,rg.vi(3):rg.vf(3))+...
    mcp.psi*gmp.dt*vrtxN(:,rg.vi(4):rg.vf(4));

coordinates=rg.vi(3):rg.vf(3);
updated(firstPoint,:)=vrtxN(edgeN{1}(:,1),coordinates);
updated(lastPoint,:)=vrtxN(edgeN{1}(:,2),coordinates);
for j=1:max(segmentCount)
    previous=firstPoint(segmentCount>=j)+j-1;
    current=previous+1;
    updated(current,:)=wrapRelative(updated(current,:),updated(previous,:),box,half,shearOffset);
end
edgeN{2}=mat2cell(updated,segmentCount+1,2);
segmentLength=sqrt(sum((updated(finish,:)-updated(start,:)).^2,2));
edgeN{4}=mat2cell(segmentLength,segmentCount,1);

[eTn,eTp]=deal(zeros(size(edge{1},1),1));
for ii=1:eMx
    [eTn(ii),eTp(ii)]=ts_edgeFixedTension(faceN,edgeN,rg,mcp,ii);
end
edgeN{1}(:,rg.ei(3))=edgeN{1}(:,rg.ei(3))...
    -gmp.dt/mcp.taut*(edgeN{1}(:,rg.ei(3))-eTn)...
    +mcp.mu/mcp.taut*sqrt(gmp.dt)/2*eTp...
    .*normrnd(0,1,eMx,1);

for ii=1:eMx
    emd=edgeN{2}{ii};
    eln=edgeN{4}{ii};
    if sum(eln)<gmp.edpc*(edgeN{1}(ii,rg.ei(4))-1/3) || ...
            sum(eln)>gmp.edpc*(edgeN{1}(ii,rg.ei(4))+4/3) || ...
            (max(eln)/min(eln)>2)
        [edgeN{2}{ii},edgeN{1}(ii,rg.ei(4))]=...
            ts_edgeMidVrtxAverage(emd,gmp.edpc);
        edgeN{4}{ii}=ts_edgeLen(edgeN{2}{ii});
    end
    edgeN{1}(ii,rg.ei(5):rg.ef(5))=ts_edgeRange(edgeN,ii);
end

for ii=1:gmp.nFa
    fEd=faceN{3}{ii};

    [faceN{1}(ii,rg.fi(2)),faceN{1}(ii,rg.fi(6):rg.ff(6))]...
        =ts_faceArea(fEd,edge{2},gmp.bs,gmp.sstn);
    faceN{1}(ii,rg.fi(3))=0;
    for jj=1:size(fEd,2)
        faceN{1}(ii,rg.fi(3))=faceN{1}(ii,rg.fi(3))+...
            sum(edge{4}{abs(fEd(jj))});
    end
    faceN{1}(ii,rg.fi(4))=...
        ts_faceNeiCount(fEd,edge{1}(:,rg.ei(2):rg.ef(2)),ii);
    faceN{1}(ii,rg.fi(5))=size(fEd,2);
end

vrtxN=ts_crdDomain(vrtxN,rg,gmp);

end

function points=wrapRelative(points,reference,box,half,shearOffset)
% Correct each point relative to its own reference in the author's order.
move=points(:,2)-reference(:,2)>half;
points(move,2)=points(move,2)-box;
points(move,1)=points(move,1)-shearOffset;
move=points(:,2)-reference(:,2)<-half;
points(move,2)=points(move,2)+box;
points(move,1)=points(move,1)+shearOffset;
move=points(:,1)-reference(:,1)>half;
points(move,1)=points(move,1)-box;
move=points(:,1)-reference(:,1)<-half;
points(move,1)=points(move,1)+box;
end
