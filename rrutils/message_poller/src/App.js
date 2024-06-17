import React, { useState, useEffect } from "react";
import axios from "axios";

const useDocumentTitle = (title) => {
  useEffect(() => {
    document.title = title;
  }, [title]);
};

function App() {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const fetchMessages = async () => {
      try {
        const response = await axios.get("/get_messages");
        const newMessages = response.data;
        for (const message of newMessages) {
          message.time = Date.now();
        }
        setMessages((prevMessages) => [...prevMessages, ...newMessages]);
      } catch (error) {
        console.error("Error fetching messages:", error);
      }
    };

    const intervalId = setInterval(fetchMessages, 100);

    return () => clearInterval(intervalId);
  }, []);

  // Group messages by ID into conversations
  const conversations = {};
  let runningCount = 0;
  let completedCount = 0;

  for (const message of messages) {
    const { id, path, prompt, responses } = message;

    if (!conversations[id]) {
      conversations[id] = { prompt: [], responses: [] };
    }

    if (path === "report_start_request") {
      if (typeof prompt === "string") {
        conversations[id].prompt = [{ role: "no_role", content: prompt }];
      } else if (Array.isArray(prompt)) {
        if (typeof prompt[0] === "string") {
          conversations[id].prompt = prompt.map((content) => ({
            role: "no_role",
            content,
          }));
        } else {
          conversations[id].prompt = prompt;
        }
      }
    } else if (path === "report_request_success") {
      conversations[id].responses = responses;
    }
  }

  // Now, determine the status of each conversation
  for (const id in conversations) {
    if (conversations[id].responses.length > 0) {
      conversations[id].status = "completed";
      completedCount++;
    } else {
      conversations[id].status = "running";
      runningCount++;
    }
  }

  const ROLE_EMOJIS = {
    user: "👤",
    assistant: "🤖",
    system: "🔧",
    no_role: "⚫",
  };

  useDocumentTitle(`(${runningCount}|${completedCount}) LM API`);

  return (
    <div className="App">
      <h1>Received Messages</h1>
      <p>Running Requests: {runningCount}</p>
      <p>Completed Requests: {completedCount}</p>
      <div>
        {Object.keys(conversations)
          .sort((a, b) => (conversations[a].status === "running" ? -1 : 1))
          .slice(0, 100)
          .map((id) => (
            <div
              key={id}
              style={{
                border: "1px solid #ccc",
                margin: "10px",
                padding: "10px",
                display: "flex",
                flexDirection: "row",
                // maxWidth: "200px",
              }}
            >
              <div>
                <p>Status: {conversations[id].status}</p>
              </div>
              <div>
                <strong>Prompt:</strong>
                <div>
                  {conversations[id].prompt.map((item, index) => (
                    <div key={index} style={{ whiteSpace: 'pre-line'}}>
                      {ROLE_EMOJIS[item.role]} {item.content}
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <strong>Responses:</strong>
                <div>
                  {conversations[id].responses.map((response, index) => (
                    <div key={index} style={{ whiteSpace: 'pre-line' }}>📝 {response.completion} </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}

export default App;
