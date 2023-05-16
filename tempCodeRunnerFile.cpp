plt::figure_size(1200, 780);
    // Add graph title
    plt::title("Sample figure");
    std::vector<double> x(trajectory.cols()),y(trajectory.cols());

    for(int i=0;i<trajectory.cols();i++){
        x[i]=trajectory.cwiseAbs()(14, i);
        y[i]=trajectory.cwiseAbs()(0, i);
    }

    plt::plot(x,y);
    const char* filename = "test.png";
    std::cout << "Saving result to " << filename << std::endl;
    plt::save(filename);