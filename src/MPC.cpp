#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
size_t N = 20;
double dt = 0.1;

// Indexes for the first step var value
// We need these indexes because all vars (for N steps) are packed together into vars[]
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;  // there are only (N-1) delta
// and of couse there are also only (N-1) a as well

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

const double ref_v = 100;  // target velocity during optimization

namespace {

double deg2rad(double x) { return x * M_PI / 180; }
double rad2deg(double x) { return x * 180 / M_PI; }

} // namespace

class FG_eval {
 public:
  // Fitted polynomial coefficients
  Eigen::VectorXd coeffs;
  FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    // Ipopt asks us to store the cost function f(x)'s value in fg[0], and constraint functions g(x)
    // _1, g(x)_2, ...'s value in fg[1], fg[2], ...

    // OK, now we set up costs

    fg[0] = 0;  // Initialize the cost

    // Set weight for different parameters
    const int cte_cost_weight = 2000;
    const int epsi_cost_weight = 2000;
    const int v_cost_weight = 1;
    const int delta_cost_weight = 10;
    const int a_cost_weight = 10;
    const int delta_change_cost_weight = 100;
    const int a_change_cost_weight = 10;

    // Add cost for cte, epsi and ev(error of velocity) for all steps
    // This is for reducing the cte, epsi and ev for all steps
    // Notice we have N calculations for each of them
    for (auto t = 0; t < N; ++t) {
      fg[0] += cte_cost_weight * CppAD::pow(vars[cte_start + t], 2);
      fg[0] += epsi_cost_weight * CppAD::pow(vars[epsi_start + t], 2);
      fg[0] += v_cost_weight * CppAD::pow(vars[v_start + t] - ref_v, 2);
    }

    // Add cost for delta and a
    // This is for reducing the steer and throttle for all actuations
    // Notice we have N-1 calculations for each of them
    for (auto t = 0; t < N - 1; ++t) {
      fg[0] += delta_cost_weight * CppAD::pow(vars[delta_start + t], 2);
      fg[0] += a_cost_weight * CppAD::pow(vars[a_start + t], 2);
    }

    // Add cost for delta change and a change
    // This is for reducing the steer change and throttle change between every adjacent actuations
    // Notice we have N-2 calculations for each of them
    for (auto t = 0; t < N - 2; ++t) {
      fg[0] += delta_change_cost_weight *
          CppAD::pow(vars[delta_start + t + 1] - vars[delta_start + t], 2);
      fg[0] += a_change_cost_weight *
          CppAD::pow(vars[a_start + t + 1] - vars[a_start + t], 2);
    }

    // OK, now we set up constraints

    // Add constraints for initial step
    // This makes sure that our planning path starts from the given initial state
    // According to the way we define constraints upper bound and lower bound,
    // we are actually writing g(x) = val here.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // Add constraints between every 2 adjacent steps
    for (auto t = 1; t < N; ++t) {
      // State at new step (at t)
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> v1 = vars[v_start + t];
      AD<double> cte1 = vars[cte_start + t];
      AD<double> epsi1 = vars[epsi_start + t];

      // State at old step (at t-1)
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      AD<double> v0 = vars[v_start + t - 1];
      AD<double> cte0 = vars[cte_start + t - 1];
      AD<double> epsi0 = vars[epsi_start + t - 1];

      // Actuator at old step (at t - 1)
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> a0 = vars[a_start + t - 1];

      // Desired y, i.e., p(x0)
      AD<double> des_y = coeffs[0]
          + coeffs[1] * x0
          + coeffs[2] * CppAD::pow(x0, 2)
          + coeffs[3] * CppAD::pow(x0, 3);

      // Desired psi, i.e., atan(p'(x0))
      AD<double> des_psi = atan(coeffs[1]
          + 2 * coeffs[2] * x0
          + 3 * coeffs[3] * pow(x0, 2));

      // Set up constraints between old and new step
      // According to the way we define constraints upper bound and lower bound,
      // we are actually writing g(x) - val = 0 here (same meaning as g(x) = val)
      // This makes sure that every step transition follow the our kinetics model
      fg[1 + x_start + t] = x1 - (x0 + v0 * CppAD::cos(psi0) * dt);
      fg[1 + y_start + t] = y1 - (y0 + v0 * CppAD::sin(psi0) * dt);
      fg[1 + psi_start + t] = psi1 - (psi0 + v0 * delta0 / Lf * dt);
      fg[1 + v_start + t] = v1 - (v0 + a0 * dt);

      fg[1 + cte_start + t] = cte1 - ((des_y - y0) + (v0 * CppAD::sin(epsi0) * dt));
      fg[1 + epsi_start + t] = epsi1 - ((des_psi - psi0) + v0 * delta0 / Lf * dt);
    }

    // Please notice that we never require (i.e., set constraints) that every step stands on the
    // poly line. Instead, we only set the constraints so that every state transition can follow
    // our kinetics model.

    // But of course, we use the poly line as the target. Why the poly line affects our
    // optimization? Because we try to optimize cte and epsi to 0, and we use the poly line to
    // help us define what cte and epsi should be. Every cte and epsi must refer to the diff
    // against the poly line!
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9

  size_t n_vars = 6 * N + 2 * (N - 1);  // number of variables in the optimization problem
  // Notice it's N-1 set of actuators, because N states only have N-1 actuations in between

  // TODO: Set the number of constraints
  size_t n_constraints = 6 * N;  // number of constraints in the optimization problem

  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.
  // Set lower and upper bounds for x, y, psi and v
  // These vars have huge reasonable value range
  for (auto i = x_start; i < delta_start; ++i) {
    vars_lowerbound[i] = -1e10;
    vars_upperbound[i] = 1e10;
  }
  // Set lower and upper bounds for delta (steer)
  // We limit delta to [-25deg, 25deg]
  // But the emulator only accept [-1, 1], so we need to normalize this type of var
  // before passing into emulator
  for (auto i = delta_start; i < a_start; ++i) {
    vars_lowerbound[i] = -deg2rad(25.0);  // in radians
    vars_upperbound[i] = deg2rad(25);  // in radians
  }
  // Set lower and upper bounds for a (throttle)
  // We limit delta to [-1, 1], as emulator requires
  for (auto i = a_start; i < n_vars; ++i) {
    vars_lowerbound[i] = -1;
    vars_upperbound[i] = 1;
  }

  // Lower and upper limits for the constraints
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; i++) {
    // Set to 0 because we write the constraint function (called  g(x)) in the format of
    // func(x+1) - func(x) = 0. The constraint value is the r-value of this equation
    // If you are still confused, Google "Getting Started With Ipopt in 90 Minutes" to understand
    // how Ipopt ask the user to write constraint function g(x)
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }
  // Initial state should have its own constraints upper and lower bounds
  // Set to specified value because the constraint function is in the format of
  // func(x) = val. The constraint value is the r-value of this equation
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;
  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.
  vector<double> solved;
  solved.push_back(solution.x[delta_start]);
  solved.push_back(solution.x[a_start]);
  for (int i = 0; i < N; ++i) {
    solved.push_back(solution.x[x_start + i]);
    solved.push_back(solution.x[y_start + i]);
  }
  return solved;
}
