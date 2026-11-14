%% Apply topological transitions.
function [vrtxN,edgeN,faceN,gmpN]=...
    ts_topTransitionType1(vrtx,edge,face,rg,gmp,mcp)

[vrtxN,edgeN,faceN,gmpN]=deal(vrtx,edge,face,gmp);

eLn=ts_edgeLenAll(edgeN,gmpN);
t1Id=find(eLn<gmp.shEd);

t1FaChk=2-sum(edgeN{1}(t1Id,rg.ei(2):rg.ef(2))==0,2);
[t1IdAll,t1ChkAll]=deal(cell(2,1));
for ii=1:2
    idTmp=find(t1FaChk==(3-ii));
    idTmp=idTmp(randperm(size(idTmp,1)));
    t1IdAll{ii}=t1Id(idTmp);
    t1ChkAll{ii}=t1FaChk(idTmp);
end

t1Id=[t1IdAll{1};t1IdAll{2}];
t1FaChk=[t1ChkAll{1};t1ChkAll{2}];

eLnc=gmp.eLnc(t1Id);
if isempty(t1Id)==0
    for jj=1:size(t1Id,1)
        if t1Id(jj)>0
            if t1FaChk(jj)==2
                chk=(sum(edgeN{4}{t1Id(jj)})<gmp.shEd && ...
                    sum(edgeN{4}{t1Id(jj)})<eLnc(jj));
            elseif t1FaChk(jj)==1
                chk=(sum(edgeN{4}{t1Id(jj)})<gmp.shEd && ...
                    sum(edgeN{4}{t1Id(jj)})<eLnc(jj));
                if chk
                    evr=edgeN{1}(t1Id(jj),rg.ei(1):rg.ef(1));
                    eeId=vrtxN(evr,rg.vi(1):rg.vf(1));
                    evId=unique(edgeN{1}(eeId(:),rg.ei(1):rg.ef(1)));
                    chk=nnz(evId~=evr(1) & evId~=evr(2))==4;
                end
            else
                chk=0;
            end
            if chk==1
                [vrtxN,edgeN,faceN]=...
                    ts_t1Transition(vrtxN,edgeN,faceN,rg,gmpN,mcp,t1Id(jj));
                efId=edgeN{1}(t1Id(jj),rg.ei(2):rg.ef(2));
                if sum(efId)==0
                    [vrtxN,edgeN,faceN,gmpN,eVl]=...
                        ts_t3ReverseTransition(vrtxN,edgeN,faceN,...
                        rg,gmpN,t1Id(jj));
                    for kk=jj:size(t1Id,1)
                        if isempty(find(eVl==t1Id(kk),1))==1
                            t1Id(kk)=-100;
                        else
                            t1Id(kk)=find(eVl==t1Id(kk));
                        end
                    end
                    clear eVl;
                end
            end
        end
    end
end

chk=(edgeN{1}(:,rg.ei(2))==0)+(edgeN{1}(:,rg.ei(2)+1)==0);
while (max(chk)==2)
    t3Id=find(chk==2);
    t3Id=t3Id(1);
    [vrtxN,edgeN,faceN,gmpN,~]=...
        ts_t3ReverseTransition(vrtxN,edgeN,faceN,rg,gmpN,t3Id);
    chk=(edgeN{1}(:,rg.ei(2))==0)+(edgeN{1}(:,rg.ei(2)+1)==0);
end

end
