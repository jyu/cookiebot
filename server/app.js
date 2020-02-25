const express = require("express");
const app = express();
const expressWs = require('express-ws')(app);
const port = 5000;

app.get('/', (req, res) => res.send("Hi Jerry\n"));

let robot_ws = null;

app.ws('/robot', function(ws, req) {
    ws.on('open', function() {
        console.log("Websocket Connection Opened with robot");
    });
    robot_ws = ws;
});

app.ws('/gestures', function(ws, req) {
    ws.on('open', function() {
        console.log("Websocket Connection Opened with cookie");
    });
    ws.on('message', function(msg) {
        console.log("Got new command <- ", msg)
        command = msg;
        if (robot_ws) {
            robot_ws.send(command);
        }  
    });
});


app.listen(port, () => console.log("CookieServer listening on port " + port));
