{% for post in site.posts %}
**[post.title }}]({{ post.permalink }})**
{{ post.excerpt}}
{% endfor %}
