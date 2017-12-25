#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
double polyeval(Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * pow(x, i);
  }
  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  // Note: Use QR decomposition to solve least square problem (which solves this polynomial fit
  // problem)
  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          vector<double> ptsx = j[1]["ptsx"];  // waypoints' x in world coordinates
          vector<double> ptsy = j[1]["ptsy"];  // waypoints' y in world coordinates
          double px = j[1]["x"];  // car's x in world coordinates
          double py = j[1]["y"];  // car's y in world coordinates
          double psi = j[1]["psi"];  // car's yaw
          double v = j[1]["speed"];  // car's speed

          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */
          // Because of the observation latency. By the time we look at the current situation, the
          // car has already moved on. So we need to do motion compensation, which is simply a
          // prediction of the state
          double dt = 0.1;  // duration of latency
          const double Lf = 2.67;  // Distance between car's front axle center to car's gravity
                                   // center

          // Use same-velocity, same-heading model for prediction within duration of this latency
          double pred_px = px + v * cos(psi) * dt;
          double pred_py = py + v * sin(psi) * dt;
          px = pred_px;
          py = pred_py;
          // psi and v won't change during this dt

          // OK, now we have the final car location px and py

          // We transform the waypoints from world coordinates into car coordinates (front is x,
          // left is y) so the polyfit result won't be something like "x = 3" (where the coefficent
          // is infinite large) because the polyfit result will almost certainly face to the
          // front along the x axis in car's coordinate and the slope won't be very large
          Eigen::VectorXd ptsx_car(ptsx.size());
          Eigen::VectorXd ptsy_car(ptsx.size());

          // For reference, remember the formula for translating from car coordinate to world
          // coordinate is:
          //   x_w = car_x + cos(psi) * x_c - sin(psi) * y_c
          //   y_w = car_y + sin(psi) * x_c + cos(psi) * y_c
          // where x_w, y_w are world coordinate, x_c, y_c are car coordinate, car_x, car_y are
          // car's location in world coordinate, and psi is counterclock-wise rotation of car's
          // heading from +x axis(i.e., yaw)
          //
          // If you use the above formula, you can get the derived formulas:
          //   x_c =  (x_w - car_x) * cos(psi) + (y_w - car_y) * sin(psi)
          //   y_c = -(x_w - car_x) * sin(psi) + (y_w - car_y) * cos(psi)
          // which lets you convert from world coordinate into car coordinate
          for (auto i = 0; i < ptsx.size(); ++i) {
            double x = ptsx[i] - px;
            double y = ptsy[i] - py;
            ptsx_car[i] =   x * cos(psi) + y * sin(psi);
            ptsy_car[i] = - x * sin(psi) + y * cos(psi);
          }

          // Now we polyfit the waypoints into a polynomial line. Polynomial line is good enough
          // since most of the road is polynomial (usually <= order of 3)
          auto coeffs = polyfit(ptsx_car, ptsy_car, 3);

          // After we have the polynomial line, we will try to optimize our actuators so our car
          // can follow this line (trajectory), comply with some other constraints, while
          // optimizing our cost (see MPC.cpp for details). So now we will prepare the
          // optimization problem input using this polynomial line

          // First get the initial cte(cross track error) and epsi(psi error). By "initial" I
          // mean the cte and epsi at the begining of the line. The initial values are part of
          // the optimization constraints we need to later provide to the solver

          // Get the initial cte(cross track error)
          double cte = polyeval(coeffs, 0) - 0;

          // Get the initial epsi
          // Let's say p(x) is the fitted polynomial line. The initial psi is the tangential
          // value at x = 0, which is p'(0), which is coeffs[1]
          double epsi = atan(coeffs[1]) - 0;

          // OK, now we will start solving the optimization problem
          Eigen::VectorXd state(6);
          state << px, py, psi, v, cte, epsi;

          // Provide initial state as one of the constraints. We will add more constraints
          // and costs later.
          // Also provide coeffs to help define costs.
          auto vars = mpc.Solve(state, coeffs);

          // TODO(jie): Get steer and throttle value from the solver result

          double steer_value;
          double throttle_value;

          json msgJson;
          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;

          //Display the MPC predicted trajectory 
          vector<double> mpc_x_vals;
          vector<double> mpc_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          msgJson["mpc_x"] = mpc_x_vals;
          msgJson["mpc_y"] = mpc_y_vals;

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;
          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(100));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
