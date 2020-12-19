import React, { Component } from 'react';
import './App.css';
import CustomTextArea from './components/CustomTextArea';
import { BrowserRouter as Router, Route} from 'react-router-dom';
class App extends Component {
  render() {
    return(
      <Router>
        <div className='App'>
          <Route path="/" exact component={CustomTextArea}/>
        </div>
      </Router>
    )

  }
}

export default App;
