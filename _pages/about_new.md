---
permalink: /
title: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .hero-animate { 
            animation: fadeInUp 0.8s ease-out; 
        }
        
        .section-animate { 
            animation: slideInLeft 0.6s ease-out;
            animation-fill-mode: forwards;
        }
        
        .award-item {
            animation: fadeInUp 0.6s ease-out forwards;
            opacity: 0;
        }
        
        .award-item:nth-child(1) { animation-delay: 0.1s; }
        .award-item:nth-child(2) { animation-delay: 0.2s; }
        .award-item:nth-child(3) { animation-delay: 0.3s; }
        .award-item:nth-child(4) { animation-delay: 0.4s; }
        
        .highlight-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .badge {
            transition: all 0.3s ease;
        }
        
        .badge:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .link-hover {
            position: relative;
            transition: all 0.3s ease;
        }
        
        .link-hover::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 0;
            height: 2px;
            background: linear-gradient(90deg, #4f46e5, #ec4899);
            transition: width 0.3s ease;
        }
        
        .link-hover:hover::after {
            width: 100%;
        }
    </style>
</head>
<body class="bg-gray-50">
    <div class="max-w-4xl mx-auto px-4 py-8">
        <!-- Hero Section -->
        <div class="hero-animate mb-12">
            <div class="bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 rounded-2xl p-8 shadow-lg">
                <h1 class="text-4xl font-bold text-gray-800 mb-6">
                    Hi there! 👋 <span class="highlight-text">Welcome to My Digital Space</span>
                </h1>
                
                <div class="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-md">
                    <h2 class="text-2xl font-semibold text-gray-800 mb-4 flex items-center">
                        <i class="fas fa-graduation-cap text-blue-600 mr-3"></i>
                        About Me
                    </h2>
                    <p class="text-lg leading-relaxed text-gray-700">
                        I'm a senior <strong class="text-blue-600">Computer Science</strong> student at 
                        <a href="https://www.zju.edu.cn/english/" class="link-hover text-blue-600 font-medium">Zhejiang University</a>, 
                        pursuing an honors degree from the 
                        <a href="http://ckc.zju.edu.cn" class="link-hover text-blue-600 font-medium">Chu Kochen Honors College</a>. 
                        Currently, I'm a research intern at <strong class="text-red-600">Berkeley AI Research (BAIR)</strong> lab, UC Berkeley, 
                        working under the guidance of 
                        <a href="https://people.eecs.berkeley.edu/~xdwang/" class="link-hover text-blue-600 font-medium">Xudong Wang</a> 
                        and <a href="https://people.eecs.berkeley.edu/~trevor/" class="link-hover text-blue-600 font-medium">Prof. Trevor Darrell</a>.
                    </p>
                </div>
            </div>
        </div>

        <!-- Research Focus Section -->
        <div class="section-animate mb-12">
            <div class="bg-white rounded-xl p-8 shadow-lg">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-microscope text-indigo-600 mr-3"></i>
                    Research Focus
                </h2>
                <p class="text-lg text-gray-700 leading-relaxed">
                    My research journey centers around <strong class="text-indigo-600">Computer Vision</strong> and <strong class="text-indigo-600">Generative AI</strong>. 
                    I'm particularly excited about building <strong class="text-purple-600">Unified Multi-modal Models</strong> that bridge the gap between text and vision. 
                    My previous work has focused on <strong class="text-pink-600">Controllable Text-to-Image Generation</strong>, including Layout-to-Image synthesis and advanced Image Editing techniques.
                </p>
            </div>
        </div>

        <!-- Awards Section -->
        <div class="section-animate mb-12">
            <div class="bg-white rounded-xl p-8 shadow-lg">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-trophy text-yellow-600 mr-3"></i>
                    Selected Honors and Awards
                </h2>
                
                <div class="space-y-4">
                    <div class="award-item flex items-start p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border-l-4 border-yellow-500">
                        <div class="badge bg-yellow-500 text-white px-3 py-1 rounded-full text-sm font-semibold mr-4 mt-1">
                            🏆 2025
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800">SenseTime Scholarship</h3>
                            <p class="text-gray-600 text-sm">Top 30 recipients annually in China</p>
                        </div>
                    </div>
                    
                    <div class="award-item flex items-start p-4 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg border-l-4 border-purple-500">
                        <div class="badge bg-purple-500 text-white px-3 py-1 rounded-full text-sm font-semibold mr-4 mt-1">
                            🥇 2022
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800">ICPC Gold Medal</h3>
                            <p class="text-gray-600 text-sm">International Collegiate Programming Contest, Shenyang Site</p>
                        </div>
                    </div>
                    
                    <div class="award-item flex items-start p-4 bg-gradient-to-r from-pink-50 to-rose-50 rounded-lg border-l-4 border-pink-500">
                        <div class="badge bg-pink-500 text-white px-3 py-1 rounded-full text-sm font-semibold mr-4 mt-1">
                            🥇 2022
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800">CCPC Gold Medal</h3>
                            <p class="text-gray-600 text-sm">China Collegiate Programming Contest, Guangzhou Site</p>
                        </div>
                    </div>
                    
                    <div class="award-item flex items-start p-4 bg-gradient-to-r from-blue-50 to-cyan-50 rounded-lg border-l-4 border-blue-500">
                        <div class="badge bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-semibold mr-4 mt-1">
                            🥇 2023/24
                        </div>
                        <div>
                            <h3 class="font-semibold text-gray-800">ZJCPC Gold Medal</h3>
                            <p class="text-gray-600 text-sm">Zhejiang Provincial Collegiate Programming Contest</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Vision Section -->
        <div class="section-animate mb-12">
            <div class="bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 rounded-xl p-8 shadow-lg">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-palette text-pink-600 mr-3"></i>
                    My Vision
                </h2>
                <blockquote class="text-xl font-medium text-gray-700 italic leading-relaxed border-l-4 border-pink-500 pl-6 bg-white/50 p-6 rounded-lg">
                    "My ultimate goal is to democratize creativity through AI - building models that can 
                    <strong class="highlight-text">Make Everybody Their Own Artist, Effortlessly</strong>."
                </blockquote>
                <p class="mt-4 text-gray-600">
                    💡 Have an exciting idea or want to explore potential collaborations? I'd love to hear from you!
                </p>
            </div>
        </div>

        <!-- Publications Section -->
        <div class="section-animate mb-12">
            <div class="bg-white rounded-xl p-8 shadow-lg">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-book-open text-green-600 mr-3"></i>
                    Publications
                </h2>
                <div class="bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-6 text-center">
                    <p class="text-gray-700 mb-4">🔬 Explore my research contributions and academic work</p>
                    <a href="https://horizonwind2004.github.io/publications/" 
                       class="inline-flex items-center px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
                        <i class="fas fa-arrow-right mr-2"></i>
                        View Full Publication List
                    </a>
                </div>
            </div>
        </div>

        <!-- Miscellaneous Section -->
        <div class="section-animate">
            <div class="bg-white rounded-xl p-8 shadow-lg">
                <h2 class="text-2xl font-bold text-gray-800 mb-6 flex items-center">
                    <i class="fas fa-star text-orange-600 mr-3"></i>
                    Miscellaneous
                </h2>
                
                <div class="space-y-6">
                    <div class="bg-gradient-to-r from-orange-50 to-yellow-50 rounded-lg p-6">
                        <p class="text-gray-700 leading-relaxed">
                            I'm an ACGN lover, so I'm enthusiastic about the Image, Video, Music and Vocal Generation, especially the <strong>model which have a good controllability</strong>.
                        </p>
                    </div>
                    
                    <div class="bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg p-6">
                        <p class="text-gray-700 leading-relaxed mb-4">
                            Previously, I've also been a member of the ZJU ACM/ICPC team, and I've reached a rating of 
                            <span class="bg-red-500 text-white px-2 py-1 rounded font-bold">2478</span> on 
                            <a href="https://codeforces.com/profile/epyset" class="link-hover text-blue-600 font-medium">Codeforces</a>. 
                            You can check my old blog <a href="https://www.luogu.com.cn/user/77426" class="link-hover text-blue-600 font-medium">here</a> 
                            where I documented my competitive programming experiences.
                        </p>
                        
                        <div class="text-center">
                            <img src="https://cfrating.baoshuo.dev/rating?username=Epyset" 
                                 alt="Codeforces Rating Graph" 
                                 class="inline-block rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
