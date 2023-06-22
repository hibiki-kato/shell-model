Eigen::VectorXi isLaminarPoint_perDim(Eigen::VectorXcd state){
// if the points is in laminar flow, return true. Otherwise return false.  state can include time bcause it will be dropped.
    for (int i = 0; i < state.size(); ++i){
        Eigen::VectorXd distance = (laminar.row(i).cwiseAbs() - state.middleRows(row_start, row_end).replicate(1, laminar.cols()).cwiseAbs()).colwise().norm();

    return (distance.array() < epsilon).any();
}