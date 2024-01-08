{% for post in site.posts %}
**[{{ post.title }}]({{ post.permalink }})**
{{ post.date}}
{{ post.excerpt}}
{% endfor %}
