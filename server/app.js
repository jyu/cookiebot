const express = require("express");
const app = express();
const expressWs = require("express-ws")(app);
const port = 5000;

app.get("/", (req, res) => res.send("CookieServer Dashboard Under Construction\n"));

let robot_ws = null;

app.ws("/robot", function(ws, req) {
    console.log("Connection established with Robot");
    robot_ws = ws;
    ws.on("message", function(msg) {
        console.log("CookieServer <- Robot: " + msg);
    });
    ws.on("close", function() {
        console.log("Connection with Robot closed");
        robot_ws = null;
    });
});

app.ws("/gestures", function(ws, req) {
    console.log("Connection established with Gestures")
    ws.on("message", function(msg) {
        console.log("CookieServer <- Gestures: " + msg)
        command = msg;
        if (robot_ws) {
            console.log("CookieServer -> Robot: " + msg)
            robot_ws.send(command);
        }  
    });
    ws.on("close", function() {
        console.log("Connection with Gestures closed");
    });
});


app.listen(port, () => console.log("CookieServer listening on port " + port));
