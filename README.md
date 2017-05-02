# Simple-DeepAPISearcher

### DeepAPISearching (with tensorflow)

> 아직 개발중...

+ <b> seq2seq 알고리즘을 이용해서 자연어를 입력하면 java API 를 출력하는 시스템 </b>

    - tensorflow
    - seq2seq
    - Deep API Learning 논문을 직접 구현해본 것

Link : [DeepAPILearning](https://arxiv.org/abs/1605.08535)


---

개발자는 해당 API를 호출하여 기존 클래스 라이브러리 또는 프레임 워크를 재사용하는 경우가 있습니다. 
어떤 API를 사용하고, 사용 순서 (API 사이의 메소드 호출 순서)를 얻는 것이이 점에서 매우 도움됩니다.
예를 들어, XML 파일을 구문 분석하는 방법을 구현하기 위해 
JDK 라이브러리를 사용하여 "XML 파일을 구문 분석"하려면 원하는 API 사용 순서는 다음과 같습니다.

---

<pre>
<code>
parse XML files => DocumentBuilderFactory.newInstance
                   DocumentBuilderFactory.newDocumentBuilder
                   DocumentBuilder.parse
</code>
</pre>

![seq2seq](http://4.bp.blogspot.com/-aArS0l1pjHQ/Vjj71pKAaEI/AAAAAAAAAxE/Nvy1FSbD_Vs/s1600/2TFstaticgraphic_alt-01.png)


> dlcjfgmlnasa@naver.com
