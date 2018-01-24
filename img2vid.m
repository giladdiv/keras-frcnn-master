function img2vid(img_folder,video_dst,fps)
    v = VideoWriter(video_dst,'Motion JPEG AVI');
    v.FrameRate = fps;
    open(v);
    d = dir([img_folder,'/*.png']);
    d = d(3:end);
    for i = 1: length(d)
        if mod(i,50)==0
           display(['finished ',num2str(i),' images']) 
        end
        img = imread(fullfile(img_folder,d(i).name));
        writeVideo(v,img)
    end
    close(v)
end