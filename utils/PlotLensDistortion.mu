PlotLensDistortion := proc(x2in=x, y2in=y, dr1=0, dr2=0, dt1=0, dt2=0, thetitle="Vector Plot")
// x2 is the expression describing the distorted x coordinate
// y2 is the expression describing the distorted y coordinate
// dr1, dr2 are the radial distortion coefficients (k1 and k2)
// dt1, dt2 are the tangential distortion coefficients (p1 and p2)

begin
  mylines := [];
  mypoints := [];
// -1 .. 1
for x from -1 to 1 step 0.2 do
    for y from -1 to 1 step 0.2 do
      xout := x2in | [x_1 = x, y_1 = y, k_1=dr1, k_2=dr2, p_1=dt1, p_2=dt2];
      yout := y2in | [x_1 = x, y_1 = y, k_1=dr1, k_2=dr2, p_1=dt1, p_2=dt2];

      startpt := [x,y];
      stoppt := [xout, yout];

      mypoint := plot::Point2d(startpt, PointStyle=FilledDiamonds, PointColor = RGB::Red);
      myline := plot::Arrow2d(startpt,stoppt,TipLength = 1);

      mylines := [mylines, myline];
      mypoints := [mypoints, mypoint];
    end_for;
  end_for;
  plot(mylines, mypoints, GridVisible = TRUE, Header=thetitle);

end_proc;