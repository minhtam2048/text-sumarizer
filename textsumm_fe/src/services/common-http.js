import axios from "axios";

const httpCommon = (payload, method = "PUT") => {
    console.log(payload);

    return axios({
        method,
        url: "http://localhost:8080/api/posts",
        headers: {
            "Content-Type": "application/json"
        },
        data: payload
    });
};

export default httpCommon;