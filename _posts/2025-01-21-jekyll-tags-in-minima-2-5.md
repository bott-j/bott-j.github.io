---
layout: post
title: Jekyll Tags in Minima 2.5
tags: 
- Jekyl 
image: /assets/posts/jekyll-tags-in-minima-2-5/jekyll-tags-in-minima-2-5.png
---
Tagging is a feature of blogs which allows posts to be grouped by a common topic. Jekyll is the static site creation software used to build this blog, and while Jekyll supports tagging the Minima 2.5 theme in use does not provide functionality to display tags. This post explains how to use custom liquid scripting to extend the Minima 2.5 HTML templates to display post tags and create a custom tag index for a blog.   

# Tagging Posts

In Jekyll, tags are added to posts in the front matter at the beginning of a post mark-down file. 

For example, the following front matter sample specifies 'Algorithms' and 'SCADA' as two related tags. 

```
{% raw %}
---
layout: post
title: Swinging Door Trending - Part 1 
tags: 
- Algorithms
- SCADA 
---
{% endraw %}
```

# Displaying Associated Tags in Posts

We can edit the post layout by creating a 'post.html' file under the '_layouts/' directory. If this directory doesn't it can be created. The new post layout we create will be used instead of the default one. When creating the 'post.html' file it is useful to start with the original file used by the Minima theme, which is available in the [Minima repository on GitHub](https://github.com/jekyll/minima/tree/2.5-stable). 

Editing the 'post.html' template file, we use add a liquid script to include a new template file called 'tags_post.html'.

For example, to display tags at the end of the page the following include can be added to the end of the content in 'post.html'. 

```
{% raw %}
{% include tags_post.html %}
{% endraw %}
```

The 'tags_post.html' file will be created in the '_includes/' directory which may be created under the Jekyll root directory.

In 'tags_post.html' we will use the below liquid script to iterate over tbe tags in the page. For each tag we will include a HTML link to the 'tags.html' tag index file explained in the next section. We will use the tag name as the identifier so that the client browser will automatically scroll to the location of the tag heading in the index. The tag name will be passed through the 'slugify' filter in order to convert the tag to a URL friendly format. 

```
{% raw %}
<span>
    Tags:&nbsp;
    {% for tag in page.tags %}
        &nbsp;<a href="/tag_index.html#{{ tag | slugify }}">[&nbsp;{{ tag }}&nbsp;]</a>&nbsp;
    {% endfor %}
</span>
{% endraw %}
```

The image below shows the above example rendered in HTML by Jekyll.

![Image of tag word links shown at end of post content.](/assets/posts/jekyll-tags-in-minima-2-5-1.png?raw=true){: width="50%"}

# Creating a Tag Index Page

We can create a tag index page using liquid scripting also. For this blog I have created a 'tags.html' file in the Jekyll root directory.

For the tag index we will need to list all of the tags used in the site. To do this we will start by creating an empty array called 'site_tags' using liquid scripting:

```
{% raw %}
{% comment %} Create an empty array. {% endcomment %} 
{% assign site_tags = "" | split: '' %}
{% endraw %}
```

The global variable site.tags is a dictionary, if we iterate over each entry we get the first element as a key which is the name of the tag. The second element as an array of posts associated with that tag.

In the following liquid script, for each item in site.tags, we push the name of the tag into our site_tags array.

```
{% raw %}
{% comment %} Populate array with tags. {% endcomment %} 
{% for tag in site.tags %}
  {% assign site_tag = tag | first %}
  {% assign site_tags = site_tags | push: site_tag %}
{% endfor %}
{% endraw %}
```

We then sort the tag name array alphabetically:

```
{% raw %}
{% comment %} Sort tags alphabetically. {% endcomment %} 
{% assign site_tags_alphabetical = site_tags | sort_natural %}
{% endraw %}
```

Following this, for each tag in the alphabetically sorted list we display it in HTML as a heading, and then further iterate over the posts under that tag which are displayed as items in an unordered list.

```
{% raw %}
{% comment %} For each tag, display posts. {% endcomment %} 
{% assign prev_tag = "" %}
{% for sorted_tag in site_tags_alphabetical %}
    {% assign tag = sorted_tag | strip %}
    {% assign current_tag = tag %}
    {% if current_tag != prev_tag %}
        <!-- Create tag pages for current character -->
        {% assign prev_tag = current_tag %}
        <!-- Create new tag page, hide by default -->
        {% assign count = site.tags[current_tag].size %}
        <h2 style="margin-top:33px" id="{{ current_tag | slugify }}">{{ current_tag }} ({{count}})</h2>
    {% endif %}
    <!-- Display Tags starting with current character -->
    <ul>
        {% for post in site.tags[tag] %}
        <li>
            <a href="{{ post.url }}">{{ post.title }}</a> ({{ post.date | date_to_string }})
        </li>
        {% endfor %}
    </ul>
    <!-- End last page -->
    {% if forloop.last %}{% endif %}
{% endfor %}
{% endraw %}
```

The following image shows the HTML output rendered by Jekyll for tag index page. 

![Image of tag word links shown at end of post content.](/assets/posts/jekyll-tags-in-minima-2-5-2.png?raw=true){: width="50%"}

# GitHub Repository

You can find the above customization compatible with the Minima 2.5 template for Jekyll in my GitHub repository [located here](https://github.com/bott-j/bott-j.github.io). 


 
