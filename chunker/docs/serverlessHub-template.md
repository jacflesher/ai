# ServerlessHub Document Template
This document is meant to be used as reference for creating documents on the ServerlessHub. 

## Table of Contents

- ### **[Code Blocks](#code-blocks-1)**
- ### **[Spacing](#spacing-1)**
- ### **[Call Outs](#call-outs-1)**
- ### **[Embedded Videos](#embedded-videos-1)**
- ### **[Tables](#tables-1)**
- ### **[Line Separation](#line-separation-1)**
- ### **[Headers](#headers-1)**
- ### **[Links](#links-1)**
- ### **[Text Styles](#text-styles-1)**
- ### **[Lists](#lists-1)**
- ### **[Inline Highlighting](#inline-highlighting-1)**
- ### **[Adding Images](#adding-images-1)**
- ### **[Tabs](#tabs-1)**

<br>

---

<br>

#### *Note

All of these examples don't have to be used individually, you can mix and match, so feel free to do so!

![](./imgs/mix-match-example.png)

<br></br>


## Code Blocks
All code blocks should follow this format:

    ```terraform linenums="1"
    startup_probe_type = "tcp"
    startup_probe_inital_delay_seconds = 0
    startup_probe_period_seconds = 240
    startup_probe_timeout_seconds = 240
    startup_probe_failure_threshold = 1
    ```

<br>

* In order to create a code block, add three back-ticks before and after the code you want to show. This is referred to as fenced code blocks.
  
* Directly after the first three back-ticks should be the programming language you are using. In this instance, this block uses 'terraform'. Supported languages can be found [here](https://www.w3schools.io/file/markdown-code-fence-blocks/#:~:text=It%20supports%20the%20following%20code%20languages.)
  
* Finally, use linenums in order to create numbered lines on your code block. In most instances the number 1 should suffice.

<br>

In the end the code block you created should look something like this:

![](./imgs/Code-Block-Example.png)

<br></br>


## Spacing

If you would lke a bit more space in between bullet points, paragraphs, etc. You can mix HTML with MarkDown like so:

    <br>

* If you need more/less space, feel free to add more ```</br>```



<br></br>

## Call-Outs
If there is a note, an example, a tip you want to emphasize, admonitions or Call-Outs are an amazing way to do so! Mkdocs offers a variety of options all with the same format like so:

    !!! note
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla et euismod nulla. Curabitur feugiat, tortor non consequat finibus, justo purus auctor massa, nec semper lorem quam in massa.

<br></br>

Here is a comprehensive list of admonitions that can be used:

   ![](./imgs/question-admonition.png)

   ![](./imgs/success-admonition.png)

   ![](./imgs/tip-admonition.png)

   ![](./imgs/info-admonition.png)

   ![](./imgs/abstract-admonition.png)

   ![](./imgs/note-admonition.png)

   ![](./imgs/example-admonition.png)

   ![](./imgs/bug-admonition.png)

   ![](./imgs/danger-admonition.png)

   ![](./imgs/failure-admonition.png)

   ![](./imgs/warning-admonition.png)

   ![](./imgs/quote-admonition.png)

* If you would like to change which admonition you use just switch the keywords. Ex: note -> abstract

<br></br>

## Embedded Videos

Most of the time, the videos we embed can be found on the [Ford Videosat Page](https://videosat.ford.com/#/). The steps to add embedded videos from this page are as follows:

1. On the right-hand side of the page, find the sharing button and click it.
    ![](./imgs/embedded-video-tutorial-1.png)

<br></br>

2. After clicking the 'Sharing' option, click the 'Embed tab and copy the text within.
   ![](./imgs/embedded-video-tutorial-2.png)

<br></br>

3. Select where the embedded video will be placed and put the iframe within a div element like so:
   
    ```html 
        <div style="text-align: center">
         <iframe width="640" height="360" src="https://videosat.ford.com/embed?id=358063e6-e25e-4c0f-863a-cc0c472fccc3" style="" frameborder="0" allowfullscreen></iframe>
        </div>
    ```

<br></br>

4. Above the embedded video (if there are multiple videos one above the first video is fine) place a call-out for users if they have trouble seeing the video like so:
   
   ```
   !!! note
       If the video below does not load make sure you're already signed into [Ford videosat](https://videosat.ford.com/#/){:target="_blank"} and refresh the page.
    ```

<br></br>

## Tables

Tables within mkdocs are formatted a bit differently than regular Markdown. You can make a table within the docs like so:

```
| Method   | Description                          |
| -------- | ------------------------------------ |
| `GET`    | :material-check:     Fetch resource  |
| `PUT`    | :material-check-all: Update resource |
| `DELETE` | :material-close:     Delete resource |
```

The ```:material-check:``` is another word for an icon or emoji that mkdocs uses. You can place any icon that makes sense with the entry. Icons supported by mkdocs can be found [here](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/).

If you wish to change the alignment of a table (left, right, center), that can be done by placing the ```:``` character on the dashes.

- Left:
  
    ```
    | Method   | Description                          |
    | :------- | :----------------------------------- |
    | `GET`    | :material-check:     Fetch resource  |
    | `PUT`    | :material-check-all: Update resource |
    | `DELETE` | :material-close:     Delete resource |
    ```

- Right:
  
    ```
    |   Method |                          Description |
    | -------: | -----------------------------------: |
    |    `GET` |  :material-check:     Fetch resource |
    |    `PUT` | :material-check-all: Update resource |
    | `DELETE` | :material-close:     Delete resource |
    ```

- Center:
  
    ```
    |  Method  |             Description              |
    | :------: | :----------------------------------: |
    |  `GET`   | :material-check:     Fetch resource  |
    |  `PUT`   | :material-check-all: Update resource |
    | `DELETE` | :material-close:     Delete resource |
    ```

<br>

Your table should end up looking something like this:

![](./imgs/table-example.png)

<br></br>

## Line separation

Besides having two hashes, there is another way to add a line of separation between one section to the next. It only requires a few dashes like so:

    ---

And it should look something like this:

---

<br></br>

## Headers

There are several ways to create headers within Markdown. However, headers need to be formatted in a specific manner.

```# Header 1```
* This header should be used at the top of the page

```## Header 2```
* This header can be used to separate sections of the page

```### Header 3```
* This header can be used to emphasize a subsection on the page

```#### Header 4```
* This header can be used to bold some text

<br></br>

## Links

Linking to a url can look messy. Instead, with MarkDown you can link with text! It can be formatted like so:

    [Cisco](www.cisco.com)

This formatting can also be applied to section headers in the same document. However, instead of a URL, a hash then the name of the section:

    [Some text](#section)

An additional way to add to links is to allow them to be opened in a different tab. You can add this functionality like so:

    [Some text](https://randomWebsite.com){:target="_blank"}

<br></br>

## Text Styles

Markdown offers a variety of ways to style your text to _accentuate_ or **stress** a point. In order to italicize some text, put underscores before and after the word or sentence:

    _sample text_

In order to bold characters add two asterisks before and after your text:

    **sample text**

<br></br>

## Lists

There are two different types of lists that can be used numerical or bulleted:

    1. Something cool
    2. Something awesome
    3. Something even cooler

    * One point
    * Another point
    * last point

<br>

And they'll look something like this:

  1. Something cool
  2. Something awesome
  3. Something even cooler

  * One point
  * Another point
  * last point
  
<br></br>

## Inline Highlighting

In order to intensify a specfic sentence or word, you can highlight them using MkDocs like so:

    {==Some text==}

And it will look something like this:

![](./imgs/highlighting-example.png)

<br></br>

## Adding Images

Adding an image is similar to adding a link, however the images need to be in the same folder as the MarkDown file. In most cases, adding a img/ folder to keep your files tidy is best practice. You can link an image from an image folder like so:


    ![](./imgs/Code-Block-Example.png)

<br></br>  

## Tabs

Tabs are an awesome way to show examples cleanly. In order to create tabs just put your header above your text like so:

    === "startup_http_header"

It should look something like this when you're done:

![](./imgs/tab-example-1.png)

You can also add tabs to the same line. In order to do this, it must be one after the other with the same spacing:

    === "HTTP Startup Probe"
    To configure the startup probe type to TCP, you must set the `startup_probe_type` variable to "tcp". Example:

    ```terraform linenums="1"
    startup_probe_type = "http"
    ```

    If you configure your Cloud Run service for a HTTP startup probe or liveness probe, you need to add an endpoint in your service code to respond to the probe. The endpoint can have whatever name you want, for example, "/health".

    === "TCP Startup Probe"
    To configure the startup probe type to TCP, you must set the `startup_probe_type` variable to "tcp". Example:

    ```terraform linenums="1"
    startup_probe_type = "tcp"
    ```

    For TCP startup probes, Cloud Run makes a TCP connection to open the TCP Socket on the specified port. By default, the TCP port is taken from what is configured in the `container_port` variable, which itself defaults to 8080. If it is required that the startup probe port be different than the `container_port`, then you will need to configure the `startup_probe_port` variable with the desired port.

It should end up looking like this:

![](./imgs/tab-example-2.png)

<br></br> 