import { message } from "antd";
import React, { Component } from "react";
import { Form, TextArea, Label, Button, Input } from "semantic-ui-react";
import httpCommon from "../services/common-http";
import Spinner from "./Spinner";
import "./custom-ui.scss";

class CustomTextArea extends Component {
  constructor(props) {
    super(props);
    this.state = {
      dataToSend: null,
      num_beam: 1,
      dataRespond: null,
      inputText: "",
      outputText: "",
      placeholderInputText: "Input news here",
      isLoading: false,
      // Card Component
      change: [],
      fileReader: new FileReader(),
    };

    // Output Text Change 
    this.handleChange = this.handleChange.bind(this);
    // Numbeam change
    this.handleNumbeamChange = this.handleNumbeamChange.bind(this);
    // Summarize Button
    this.handleSubmit = this.handleSubmit.bind(this);
    // File Upload
    this.handleFileRead = this.handleFileRead.bind(this);
    this.handleFileChosen = this.handleFileChosen.bind(this);
  }

  handleChange(event) {
    this.setState({ inputText: event.target.value });
  }

  handleNumbeamChange(event) {
    this.setState({ num_beam: event.target.value });
  }


  handleFileRead() {
    const content = this.state.fileReader.result;
    console.log(content);
    this.setState({
      inputText: content
    })
  }

  handleFileChosen(file) {
    this.state.fileReader.onloadend = this.handleFileRead;
    this.state.fileReader.readAsText(file);
  };

  async handleSubmit(e) {
    // send text and get predict text
    e.preventDefault();
    var dataToSend = null;

    console.log(this.state.inputText, this.state.num_beam);


    if(this.state.inputText.trim() === "") {
      message.error("Please input some text");
    } else {
      this.setState({ isLoading: true })
      var obj = { content: this.state.inputText, num_beam: this.state.num_beam }

      dataToSend = JSON.stringify(obj)

      let {data} = await httpCommon(dataToSend);
      console.log(data)

 

      this.setState({
        outputText: data.content,
        isLoading: false,
        change: [...this.state.change, data]
      });
    }
  }

  render() {
    return (
      <React.Fragment>
        <div className="container">
          <div className="row">
            <div className="col-xs-12 input-area">
              <h1>Text Summarizer</h1>
            </div>

            <div className="input-area">
              <Label>File upload</Label>
              <Input type='file' 
                  id='file'
                  accept='.txt' 
                  onChange={e => this.handleFileChosen(e.target.files[0])}
              />
              <Label>num beam</Label>
              <Input type="number" 
                     min="1" 
                     max="10" 
                     value={this.state.num_beam} 
                     onChange={this.handleNumbeamChange}
              />
            </div>
            <div style={{ display: "flex", justifyContent: 'space-around' }}>
            
              <div className="col-xs-12 col-sm-6 input-area custom-input">
                
                <Form>
                  <Label>Input text</Label>
                  <TextArea value={this.state.inputText}
                            onChange={this.handleChange} 
                            placeholder={this.state.placeholderInputText} 
                            spellCheck="false"
                            className="custom-input-text-area" />
                </Form>
              </div>
              <div className="custom-button">
                {
                  this.state.isLoading === true ? <Spinner/>:
                  <Button primary onClick={this.handleSubmit}> Summarize </Button>
                }
              </div>
              <div className="col-xs-12 col-sm-6 input-area custom-input">
                <Form>
                  { this.state.change.map(
                    (data, index) => (
                      <React.Fragment key={index} className="input-area">
                        <Label content={'num_beam: ' + data.num_beam} />
                        <TextArea value={data.content} 
                                  spellCheck="false" 
                                  rows="4"
                                  className="custom-output-text-area"/>
                      </React.Fragment>
                    )
                  )
                  }
                </Form> 
              </div>
            </div>
          </div>
        </div>
      </React.Fragment>
    );
  }
}

export default CustomTextArea;
