%% Find vertex coordinates of a given face

function fCrd=ts_faceVrtxCrd(fEd,eMd,bs,sstn)

fCrd=cell(size(fEd,2),1);
for ii=1:size(fEd,2)
    if fEd(ii)>0
        fCrd{ii}=eMd{fEd(ii)}(1:end-1,:);
    else
        fCrd{ii}=flipud(eMd{-fEd(ii)}(2:end,:));
    end
end
fCrd=vertcat(fCrd{:});
fCrd=ts_crdLocal(fCrd,bs,sstn);

end
