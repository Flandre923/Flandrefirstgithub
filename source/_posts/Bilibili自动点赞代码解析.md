---
title: Bilibili自动点赞代码解析
date: 2023-11-18 10:23:09
tags:
- bilibili
- 自动点赞
- javascript
cover: https://view.moezx.cc/images/2021/02/08/44d079df65f53f100393d8913c357102.jpg
---

## 原地址

[BiliBili_Auto_Like/autoLike.js at main · howxcheng/BiliBili_Auto_Like (github.com)](https://github.com/howxcheng/BiliBili_Auto_Like/blob/main/autoLike.js)

## 原作者

[howxcheng (程昊) (github.com)](https://github.com/howxcheng)

## 代码解析

```javascript
// ==UserScript==
// @name           Bilibili自动点赞
// @name-en        Bilibili_Auto_Like
// @namespace      http://tampermonkey.net/
// @version        2.1
// @description    哔哩哔哩视频、番剧自动点赞
// @author         Howxcheng
// @match          *://*.bilibili.com/video/*
// @match          *://*.bilibili.com/bangumi/*
// @homepageURL    https://github.com/howxcheng/BiliBili_Auto_Like
// @supportURL     https://github.com/howxcheng/BiliBili_Auto_Like/issues
// @icon           https://t1.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=http://bilibili.com&size=16
// @license        MIT
// @run-at         document-start
// @grant          unsafeWindow
// @grant          GM_xmlhttpRequest
// @grant          GM_getResourceText
// @grant          GM_notification
// @grant          GM_openInTab
// @grant          GM_getValue
// @grant          GM_setValue
// @grant          GM_addStyle
// ==/UserScript==

(function () {
  "use strict";
  var WIDE_MODE_SWITCH = false; // 是否启用<自动宽屏模式>,true:开启,false:关闭
  var LIKE_TIME_OUT = 0; // 延迟点赞时间,单位:毫秒

  var originUrl = document.location.toString();
  var like_lock = false; // 点赞计时器锁
  var like_timer = null; // 点赞计时
  var like_count = 0; // 点赞失败计数器
  var wide_lock = false; // 宽屏计时器锁
  var wide_timer = null; // 宽屏计时器锁
  var wide_count = 0; // 宽屏失败计数器
  var main_timer = null;
  var main_lock = true;
  main_timer = setInterval(changeEvent, 1000);  // setInterval是一个函数，可以每隔一定的时间执行一个函数。这里是每隔1000毫秒（1秒）执行changeEvent函数，用于检测页面是否加载完成。
  document.addEventListener("click", (_e) => {//可以在文档上添加一个事件处理器。这里是在文档上添加一个点击事件处理器，当用户点击文档时，执行一个匿名函数。这个匿名函数使用了箭头函数的语法，可以简化函数的定义。这个匿名函数的参数是_e，表示事件对象，但是这里没有用到，所以用下划线开头表示忽略。
    // console.log("监控到点击事件");
    setTimeout(() => {
        //这个匿名函数的内容是使用setTimeout函数延迟500毫秒（0.5秒）执行另一个匿名函数。这个匿名函数的内容是获取当前的网址，和原始的网址比较，如果不同，说明用户点击了其他的视频或番剧，那么就需要重新执行changeEvent函数，所以就设置一个新的计时器，每隔500毫秒执行一次changeEvent函数，并且把main_lock变量设为true，表示主计时器被锁定，避免重复设置计时器。
      var currentUrl = document.location.toString();
      if (currentUrl !== originUrl) {
        // console.log("url不同，执行操作");
        if (!main_lock) {
          main_lock = true;
          main_timer = setInterval(changeEvent, 500);
        }
      }
    }, 500);
  });
    //定义了changeEvent函数，用于检测页面是否加载完成，以及是否需要执行点赞和宽屏的操作。
  function changeEvent() {
      //changeEvent函数的内容是判断文档的状态是否为"complete"，表示文档已经加载完成，可以进行操作。如果是，那么就判断like_lock和wide_lock变量是否为false，表示点赞和宽屏的计时器是否被锁定。
    if (document.readyState === "complete") {
      // console.log("执行");
      if (!like_lock) {
        like_lock = true;
        // console.log("like锁定");
        like_count = 0;
        like_timer = setInterval(clickLike, 500);
      }
      if (WIDE_MODE_SWITCH && !wide_lock) {
         //如果没有被锁定，那么就设置相应的计时器，每隔500毫秒执行clickLike和setWideMode函数，用于实现点赞和宽屏的功能，并且把相应的锁和计数器设为true和0，表示开始执行操作。
        wide_lock = true;
        // console.log("wide锁定");
        wide_count = 0;
        wide_timer = setInterval(setWideMode, 500);
      }
        //然后清除主计时器，更新原始的网址，把main_lock变量设为false，表示主计时器被解锁，可以重新设置。
      clearInterval(main_timer);
      originUrl = document.location.toString();
      // console.log("timer解锁");
      main_lock = false;
    }
  }
  // 自动宽屏模式
  function setWideMode() {
    wide_count++;// 计数器++ 
    var _set_wide_mode_button = document.querySelector('div[class="bpx-player-ctrl-btn bpx-player-ctrl-wide"]');// 获得宽频按钮
    if (_set_wide_mode_button !== null) {//如果宽屏按钮不为空
      try {
        _set_wide_mode_button.click();//尝试点击按钮
      } catch (error) {
        // console.log(error);
      }
      // console.log("非宽屏，切换宽屏,次数：" + wide_count);
      wide_count = 64;// 计数器设置为64
    }
    if (document.querySelector('div[class="bpx-player-ctrl-btn bpx-player-ctrl-wide bpx-state-entered"]') !== null) {//如果按钮不存在
      // console.log("宽屏，跳过,次数：" + wide_count);
      wide_count = 64;//计数器设置为64
    }
    if (wide_count <= 60) return;//如果计数器小于60，宽屏按钮存在，但是为点击到,则返回
    // console.log("wide解锁");
    clearInterval(wide_timer);//清除wide_timer计数器
    goToSuitable();//滚到合适位置
    wide_lock = false;//宽屏锁关闭
  }
  // 滚动至合适位置
  function goToSuitable() {
    setTimeout(function () {
      window.scrollTo({
        top: 92,
        behavior: "smooth",
      });
    }, 1000);
  }
  // 点赞
  function clickLike() {
    like_count++;//点赞计数器
    var _like_button = document.querySelector("div[class='video-like video-toolbar-left-item']");//获得点赞按钮
    if (_like_button !== null) { // 点赞按钮存在
      try {
        // // console.log("正在点赞");
        new Promise((resolve) =>
          setTimeout(() => {
            _like_button.click();
            Toast("已自动点赞", 3000);
          }, LIKE_TIME_OUT) // 阻塞，延迟LIKE_TIME_OUT时间点赞执行，并提示信息。默认为0秒
        );
      } catch (error) {
        // console.log(error);
      }
      // console.log("未点赞，正在点赞,次数：" + like_count);
      like_count = 64; // 点赞成功 设置计数器64
    }
    if (document.querySelector("div[class='video-like video-toolbar-left-item on']") !== null) {// 获得点赞按钮失败
      // console.log("已点赞,次数：" + like_count);
      like_count = 64;//设置计数器64
    }
    if (like_count <= 60) return; // 如果点赞小于64则说明存在点赞按钮未点击到，500ms后重新点赞
    // console.log("like解锁");
    clearInterval(like_timer);//点赞成功，清楚计数器
    like_lock = false;
  }
  //界面toast提示
  function Toast(msg, duration) {
    duration = isNaN(duration) ? 3000 : duration;
    var m = document.createElement("div");
    m.innerHTML = msg;
    m.style.cssText =
      "font-family:siyuan;max-width:60%;min-width: 150px;padding:0 14px;height: 40px;color: rgb(255, 255, 255);line-height: 40px;text-align: center;border-radius: 4px;position: fixed;top: 10%;left: 50%;transform: translate(-50%, -50%);z-index: 999999;background: rgba(0, 0, 0,.7);font-size: 16px;";
    document.body.appendChild(m);
    setTimeout(function () {
      var d = 0.5;
      m.style.webkitTransition = "-webkit-transform " + d + "s ease-in, opacity " + d + "s ease-in";
      m.style.opacity = "0";
      setTimeout(function () {
        document.body.removeChild(m);
      }, d * 1000);
    }, duration);
  }
})();
```

